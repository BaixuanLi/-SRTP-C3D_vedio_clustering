class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = 'C:/LearningSoftware/DataSet/UCF101_video'

            # Save preprocess data into output_dir
            output_dir = 'C:/Users/86136/Desktop/pre_test'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return './pretrained_model/c3d-pretrained.pth'
