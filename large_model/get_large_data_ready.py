import numpy as np
import random
import sys
import os




if __name__ == '__main__':
    # define necessary parameters
    n_lag = 40
    n_pred = 1
    task_dim = [5, 5]
    target_var = 'DUEXTTAU'
    test_ratio = 0.1
    random_select = False

    data_cache_path = sys.argv[1]
    #AFG_only = True if sys.argv[4] == 'AFG' else False
    for season in [1, 2, 3, 4]:
        # season subset
        year1_avlb_days = list(
            range((season - 1) * 91 + n_lag if (season - 1) * 91 - 45 < 0 else (season - 1) * 91 + n_lag - 45,
                  season * 91 if season != 4 else season * 91 + 1))
        year2_avlb_days = list(range(365 + (season - 1) * 91 + n_lag - 45,
                                     365 + season * 91 if season != 4 else 365 + season * 91 + 1))
        avlb_days = year1_avlb_days + year2_avlb_days

        # train/test split
        # get all available days
        # random split into train/test sets
        if random_select:
            avlb_days = set(avlb_days)
            test_set = set(random.sample(avlb_days, int(len(avlb_days) * test_ratio)))
            train_set = avlb_days - test_set
        else:
            test_n = int(len(avlb_days)*test_ratio)
            test_set = avlb_days[-test_n:]
            train_set = avlb_days[:-test_n]

        # create season dir if not exist
        season_path = os.path.join(data_cache_path, 'Season' + str(season))
        if not os.path.exists(season_path):
            os.mkdir(season_path)

        # save test date
        np.save(os.path.join(season_path, 'test_days.npy'), np.array(list(test_set)))
        np.save(os.path.join(season_path, 'train_days.npy'), np.array(list(train_set)))
        # create area dir and data
        for area in [0, 1]:
            # create area dir
            area_path = os.path.join(season_path, 'Area' + str(area))
            if not os.path.exists(area_path):
                os.mkdir(area_path)




