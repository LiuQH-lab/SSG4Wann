import os
import numpy as np 
import pandas as pd
import time

class hr:
    def __init__(self, cwd, seed,  NONCOLLINEAR_channel: bool, hr4trans=None):
        self.NONCOLLINEAR_channel = NONCOLLINEAR_channel
        self.hr4trans = hr4trans
        self.paths = {}
        if hr4trans is not None:
            self.paths['nc'] = hr4trans
            print(f"loading: {self.paths['nc']}")
        else:
            if self.NONCOLLINEAR_channel:      
                self.paths['nc'] = os.path.join(cwd, seed + '_hr.dat')
                print(f"loading: {self.paths['nc']}")
            else:
                self.paths['up'] = os.path.join(cwd, seed + '.up_hr.dat')
                self.paths['dn'] = os.path.join(cwd, seed + '.dn_hr.dat')
                print(f"loading: {self.paths['up']} and {self.paths['dn']}")

        self.raw_data_dict = {}
        self.num_wann = None
        self.nrpts = None
        self.rawload()
        # print(f"finished: num_wann={self.num_wann}, nrpts={self.nrpts}")

    def rawload(self):
        if self.NONCOLLINEAR_channel or self.hr4trans is not None:
                df, self.num_wann, self.nrpts = self.raw_read(self.paths['nc'])
                self.raw_data_dict['nc'] = df
        else:
            df_up, self.num_wann, self.nrpts = self.raw_read(self.paths['up'])
            df_dn, _, _ = self.raw_read(self.paths['dn'])
            self.raw_data_dict['up'] = df_up
            self.raw_data_dict['dn'] = df_dn

    @staticmethod
    def raw_read(filepath):

        start_time = time.time()
        
        with open(filepath, 'r') as f:
            header = f.readline()
            num_wann = int(f.readline().strip())
            nrpts = int(f.readline().strip())
            
            deg_lines = int(np.ceil(nrpts / 15.0))
            degeneracies = []
            for _ in range(deg_lines):
                degeneracies.extend([int(x) for x in f.readline().split()])
                
        skip_lines = 3 + deg_lines
        
        df = pd.read_csv(
            filepath, 
            sep=r'\s+', 
            skiprows=skip_lines, 
            header=None, 
            names=['R1', 'R2', 'R3', 'i', 'j', 'Re', 'Im'],
            engine='c'
        )
        

        omega_array = np.repeat(degeneracies, num_wann**2)
        omega_array = omega_array[:len(df)]
    
        df['H'] = (df['Re'] + 1j * df['Im']) / omega_array
        
        
        df.drop(columns=['Re', 'Im'], inplace=True)
        
        end_time = time.time()
        print(f"finish loading [{os.path.basename(filepath)}] , {end_time - start_time:.2f} seconds used") 
        
        return df, num_wann, nrpts


    def hr_entry(self):
        H_dict = {}
        
        if self.NONCOLLINEAR_channel or self.hr4trans is not None:
            df = self.raw_data_dict['nc']
            grouped = df.groupby(['R1', 'R2', 'R3'])
            
            for R_tuple, group in grouped:
                Rtu = (int(R_tuple[0]), int(R_tuple[1]), int(R_tuple[2]))
                H_dict[Rtu] = {
                    "Rvec": np.array(Rtu).reshape(3, 1)
                }
                for i, j, H in zip(group['i'], group['j'], group['H']):
                    H_dict[Rtu][int(i), int(j)] = H
                    
        else:
            for spin in ['up', 'dn']:
                df = self.raw_data_dict[spin]
                grouped = df.groupby(['R1', 'R2', 'R3'])
                
                for R_tuple, group in grouped:
                    Rtu = (int(R_tuple[0]), int(R_tuple[1]), int(R_tuple[2]))
                    
                    if Rtu not in H_dict:
                        H_dict[Rtu] = {
                            "Rvec": np.array(Rtu).reshape(3, 1)
                        }
                        
                    for i, j, H in zip(group['i'], group['j'], group['H']):
                        i, j = int(i), int(j)
                        if (i, j) not in H_dict[Rtu]:
                            H_dict[Rtu][i, j] = {}
                        H_dict[Rtu][i, j][spin] = H

        self.raw_data_dict.clear() 
        return H_dict, self.num_wann
    
   
    def Kpoints_gen(self, bands_num_points, kpath, permuK):

        current_dist = 0.0
        tot = [] 
        k_labels = []
        fullk = []
        for seg in kpath:
            start = seg['start']
            end = seg['end']
            label = seg['label_start']


            segment_k = [start + (end - start) * i / (bands_num_points - 1 ) for i in range(bands_num_points)]
            fullk.extend(segment_k)
            dist_segment = np.linalg.norm((end - start) @ permuK) 
            

            segment_x = np.linspace(current_dist, current_dist + dist_segment, bands_num_points, endpoint=True)
            tot.extend(segment_x)
            
            k_labels.append((current_dist, label))
            current_dist += dist_segment
            

        k_labels.append((current_dist, kpath[-1]['label_end']))
        
        return np.array(fullk), np.array(tot), k_labels
    @staticmethod
    def convert(H_dict, num_wann, spin_channel=None):

        matrix_hr = {}
        for Rtu, block in H_dict.items():
            mat = np.zeros((num_wann, num_wann), dtype=complex)
            for key, val in block.items():
                if isinstance(key, tuple):  
                    i, j = key
                    idx_i, idx_j = i - 1, j - 1
                    
                    if isinstance(val, dict):
                        s = spin_channel if spin_channel else list(val.keys())[0]
                        mat[idx_i, idx_j] = val[s]
                    else:
                        mat[idx_i, idx_j] = val
                        
            matrix_hr[Rtu] = {
                "Rvec": block["Rvec"],
                "mat": mat
            }
        return matrix_hr
    
    @staticmethod
    def Hk_gen(matrix_hr, num_wann, kpoint, permuK, permutation):

        Hk = np.zeros((num_wann, num_wann), dtype=complex)
        
        kcart = kpoint @ permuK
        
        
        for Rtu, block in matrix_hr.items():
            Rvec = block["Rvec"].flatten()
            Rcart = Rvec @ permutation.T
            phase = np.exp(1j * np.dot(kcart, Rcart))
            Hk += block["mat"] * phase
            
        return Hk

    @staticmethod
    def hr2bds(kpoint, num_wann, hr_entry, permuK, permutation):
        matrix_hr = hr.convert(hr_entry, num_wann)
        Hk = hr.Hk_gen(matrix_hr, num_wann, kpoint, permuK, permutation)
        eigenvalues = np.linalg.eigvalsh(Hk)
        return eigenvalues


    @staticmethod
    def hrdiff(hr_identity, hr_op, index, num_wann):
        h1 = hr.convert(hr_identity, num_wann)
        h2 = hr.convert(hr_op, num_wann)
        checkpass = True
        for Rtu in h1:
            if Rtu not in h2:
                raise ValueError(f"Rtu {Rtu} is missing in this hr_op!")
            mat1 = h1[Rtu]["mat"]
            mat2 = h2[Rtu]["mat"]
            diff_max = np.max(np.abs(mat1 - mat2))
            if diff_max > 0.5:
                print(f"Warning: Max difference for Rtu {Rtu} is {diff_max:.2e}, check this operator!")
                checkpass = False
        if checkpass:
            print(f"hrdiff check passed for operator index {index}!")





        


        

