class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = 'D:/UCF-101/UCF101_video/'

            # Save preprocess data into output_dir
            output_dir = 'D:/UCF-101/preprocess_for_test/'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return './pretrained_model/c3d-pretrained.pth'
