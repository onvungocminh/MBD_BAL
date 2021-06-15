
# Config for training: dataset and their directories
# 
config = {
    'ISBI':  {'data_root': '/lrde/image/CV_2021_yizi/ISBI2013_data/',
                'data_lst': 'train_pair.lst',
                'mean_bgr': [128.85538516998292]},
    'CREMI': {'data_root': '/lrde/image/CV_2021_yizi/CREMI/',
             'data_lst': 'train_pair.lst',
             'mean_bgr': [125.65682009887695]},  
    'ISBI2012': {'data_root': '/lrde/image/CV_2021_yizi/ISBI2012/',
             'data_lst': 'train_pair.lst',
             'mean_bgr': [126.16321881612141]},   
}
config_test = {
    'ISBI':  {'data_root': '/lrde/image/CV_2021_yizi/ISBI2013_data/',
                'data_lst': 'test.lst',
                'mean_bgr': [128.85538516998292]}
}
config_val = {
    'ISBI':  {'data_root': '/lrde/image/CV_2021_yizi/ISBI2013_data/',
            'data_lst': 'val_pair.lst',
            'mean_bgr': [128.85538516998292]},
}
config_all = {
    'ISBI':  {'data_root': '/lrde/image/CV_2021_yizi/ISBI2013_data/',
            'data_lst': 'all_pair.lst',
            'mean_bgr': [128.85538516998292]},
    'CREMI': {'data_root': '/lrde/image/CV_2021_yizi/CREMI/',
             'data_lst': 'all_pair.lst',
             'mean_bgr': [125.65682009887695]},  
    'ISBI2012': {'data_root': '/lrde/image/CV_2021_yizi/ISBI2012/',
             'data_lst': 'all_pair.lst',
             'mean_bgr': [126.16321881612141]}, 
}
