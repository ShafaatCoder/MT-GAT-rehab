# # import pandas as pd
# # import numpy as np
# # from sklearn.preprocessing import StandardScaler
# # from scipy import signal
# # from IPython.core.debugger import set_trace
# # from util.joint_angle_feature import extract_joint_angles, JOINT_ANGLE_PAIRS
# # index_Spine_Base=0
# # index_Spine_Mid=4
# # index_Neck=8
# # index_Head=12   # no orientation
# # index_Shoulder_Left=16
# # index_Elbow_Left=20
# # index_Wrist_Left=24
# # index_Hand_Left=28
# # index_Shoulder_Right=32
# # index_Elbow_Right=36
# # index_Wrist_Right=40
# # index_Hand_Right=44
# # index_Hip_Left=48
# # index_Knee_Left=52
# # index_Ankle_Left=56
# # index_Foot_Left=60  # no orientation    
# # index_Hip_Right=64
# # index_Knee_Right=68
# # index_Ankle_Right=72
# # index_Foot_Right=76   # no orientation
# # index_Spine_Shoulder=80
# # index_Tip_Left=84     # no orientation
# # index_Thumb_Left=88   # no orientation
# # index_Tip_Right=92    # no orientation
# # index_Thumb_Right=96  # no orientation

# # class Data_Loader():
# #     def __init__(self, dir):
# #         self.num_repitation = 5
# #         self.num_channel = 3
# #         self.dir = dir
# #         self.body_part = self.body_parts()       
# #         self.dataset = []
# #         self.sequence_length = []
# #         self.num_timestep = 100
# #         self.new_label = []
# #         self.train_x, self.train_y= self.import_dataset()
# #         self.batch_size = self.train_y.shape[0]
# #         self.num_joints = len(self.body_part)
# #         self.sc1 = StandardScaler()
# #         self.sc2 = StandardScaler()
# #         self.scaled_x, self.scaled_y = self.preprocessing()
                
# #     def body_parts(self):
# #         body_parts = [index_Spine_Base, index_Spine_Mid, index_Neck, index_Head, index_Shoulder_Left, index_Elbow_Left, index_Wrist_Left, index_Hand_Left, index_Shoulder_Right, index_Elbow_Right, index_Wrist_Right, index_Hand_Right, index_Hip_Left, index_Knee_Left, index_Ankle_Left, index_Foot_Left, index_Hip_Right, index_Knee_Right, index_Ankle_Right, index_Ankle_Right, index_Spine_Shoulder, index_Tip_Left, index_Thumb_Left, index_Tip_Right, index_Thumb_Right
# # ]
# #         return body_parts
    
# #     def import_dataset(self):
# #         train_x = pd.read_csv("./" + self.dir+"/Train_X.csv", header = None).iloc[:,:].values
# #         train_y = pd.read_csv("./" + self.dir+"/Train_Y.csv", header = None).iloc[:,:].values
# #         return train_x, train_y
            
# #     def preprocessing(self):
# #         X_train = np.zeros((self.train_x.shape[0],self.num_joints*self.num_channel)).astype('float32')
# #         for row in range(self.train_x.shape[0]):
# #             counter = 0
# #             for parts in self.body_part:
# #                 for i in range(self.num_channel):
# #                     X_train[row, counter+i] = self.train_x[row, parts+i]
# #                 counter += self.num_channel 
        
# #         y_train = np.reshape(self.train_y,(-1,1))
# #         X_train = self.sc1.fit_transform(X_train)         
# #         y_train = self.sc2.fit_transform(y_train)
        
# #         X_train_ = np.zeros((self.batch_size, self.num_timestep, self.num_joints, self.num_channel))
        
# #         for batch in range(X_train_.shape[0]):
# #             for timestep in range(X_train_.shape[1]):
# #                 for node in range(X_train_.shape[2]):
# #                     for channel in range(X_train_.shape[3]):
# #                         X_train_[batch,timestep,node,channel] = X_train[timestep+(batch*self.num_timestep),channel+(node*self.num_channel)]
        
                        
# #         X_train = X_train_                
# #         return X_train, y_train

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from util.joint_angle_feature import extract_joint_angles, JOINT_ANGLE_PAIRS

# # KIMORE joint index positions in Train_X.csv (step of 4: x,y,z,confidence)
# index_list = [
#     0, 4, 8, 12, 16, 20, 24, 28,
#     32, 36, 40, 44, 48, 52, 56, 60,
#     64, 68, 72, 76, 80, 84, 88, 92, 96
# ]

# class Data_Loader():
#     def __init__(self, dir):
#         self.dir = dir
#         self.num_timestep = 100
#         self.num_channel = 3 + len(JOINT_ANGLE_PAIRS)  # 3 for x,y,z + angle channels
#         self.body_part = self.body_parts()
#         self.num_joints = len(self.body_part)
#         self.train_x, self.train_y = self.import_dataset()
#         self.batch_size = self.train_y.shape[0]
#         self.sc1 = StandardScaler()
#         self.sc2 = StandardScaler()
#         self.scaled_x, self.scaled_y = self.preprocessing()

#     def body_parts(self):
#         return index_list

#     def import_dataset(self):
#         x = pd.read_csv(f"./{self.dir}/Train_X.csv", header=None).values
#         y = pd.read_csv(f"./{self.dir}/Train_Y.csv", header=None).values
#         return x, y

#     def preprocessing(self):
#         samples = self.train_x.shape[0] // self.num_timestep
#         X_pos = np.zeros((samples, self.num_timestep, self.num_joints, 3))

#         for s in range(samples):
#             for t in range(self.num_timestep):
#                 for j, idx in enumerate(self.body_part):
#                     X_pos[s, t, j, :] = self.train_x[s * self.num_timestep + t, idx:idx + 3]

#         # Normalize position data
#         X_reshaped = X_pos.reshape(samples * self.num_timestep, -1)
#         X_scaled = self.sc1.fit_transform(X_reshaped).reshape(samples, self.num_timestep, self.num_joints, 3)

#         # Extract and normalize joint angles
#         X_flat = self.train_x.reshape(samples, self.num_timestep, -1)
#         joint_angles = np.array([extract_joint_angles(X_flat[s]) for s in range(samples)])
#         joint_angles = self.sc1.fit_transform(joint_angles.reshape(-1, joint_angles.shape[-1])).reshape(samples, self.num_timestep, -1)

#         # Expand and append angle features
#         angle_expanded = np.repeat(joint_angles[:, :, np.newaxis, :], self.num_joints, axis=2)
#         X_combined = np.concatenate([X_scaled, angle_expanded], axis=-1)

#         # Normalize labels
#         y = self.train_y.reshape(-1, 1)
#         y_scaled = self.sc2.fit_transform(y)

#         return X_combined, y_scaled

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from util.joint_angle_feature import extract_joint_angles, JOINT_ANGLE_PAIRS
from util.joint_distance_feature import compute_distances, DISTANCE_PAIRS

# KIMORE joint index positions in Train_X.csv (step of 4: x,y,z,confidence)
index_list = [
    0, 4, 8, 12, 16, 20, 24, 28,
    32, 36, 40, 44, 48, 52, 56, 60,
    64, 68, 72, 76, 80, 84, 88, 92, 96
]

class Data_Loader():
    def __init__(self, dir):
        self.dir = dir
        self.num_timestep = 100
        self.num_channel = 3 + len(JOINT_ANGLE_PAIRS) + len(DISTANCE_PAIRS)
        self.body_part = self.body_parts()
        self.num_joints = len(self.body_part)
        self.train_x, self.train_y = self.import_dataset()
        self.batch_size = self.train_y.shape[0]
        self.sc1 = StandardScaler()
        self.sc2 = StandardScaler()
        self.scaled_x, self.scaled_y = self.preprocessing()

    def body_parts(self):
        return index_list

    def import_dataset(self):
        x = pd.read_csv(f"./{self.dir}/Train_X.csv", header=None).values
        y = pd.read_csv(f"./{self.dir}/Train_Y.csv", header=None).values
        return x, y

    def preprocessing(self):
        samples = self.train_x.shape[0] // self.num_timestep
        X_pos = np.zeros((samples, self.num_timestep, self.num_joints, 3))

        for s in range(samples):
            for t in range(self.num_timestep):
                for j, idx in enumerate(self.body_part):
                    X_pos[s, t, j, :] = self.train_x[s * self.num_timestep + t, idx:idx + 3]

        # Normalize position data
        X_reshaped = X_pos.reshape(samples * self.num_timestep, -1)
        X_scaled = self.sc1.fit_transform(X_reshaped).reshape(samples, self.num_timestep, self.num_joints, 3)

        # Extract and normalize joint angles
        X_flat = self.train_x.reshape(samples, self.num_timestep, -1)
        joint_angles = np.array([extract_joint_angles(X_flat[s]) for s in range(samples)])
        joint_angles = self.sc1.fit_transform(joint_angles.reshape(-1, joint_angles.shape[-1])).reshape(samples, self.num_timestep, -1)
        angle_expanded = np.repeat(joint_angles[:, :, np.newaxis, :], self.num_joints, axis=2)

        # Extract and normalize inter-joint distances
        distances = np.array([compute_distances(X_flat[s]) for s in range(samples)])
        distances = self.sc1.fit_transform(distances.reshape(-1, distances.shape[-1])).reshape(samples, self.num_timestep, -1)
        dist_expanded = np.repeat(distances[:, :, np.newaxis, :], self.num_joints, axis=2)

        # Concatenate all features
        X_combined = np.concatenate([X_scaled, angle_expanded, dist_expanded], axis=-1)

        # Normalize labels
        y = self.train_y.reshape(-1, 1)
        y_scaled = self.sc2.fit_transform(y)

        return X_combined, y_scaled