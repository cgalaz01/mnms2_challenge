from sklearn.model_selection import ParameterGrid

from tensorboard.plugins.hparams import api as hp

# Sortable version of HParam
class HParamS(hp.HParam):
    
    def __init__(self, name, domain=None, display_name=None, description=None):
        hp.HParam.__init__(self, name, domain, display_name, description)
        
    def __lt__(self, other):
        return self.name.lower() < other.name.lower()


class HyperParameters():
    
    def __init__(self, search_type: str):
        # TODO: Load from file rather than hard-coded in this file
        self.HP_FLOATING_POINT = HParamS('floating_point', hp.Discrete(['16']))
        self.HP_EPOCHS = HParamS('epochs', hp.Discrete([300]))
        self.HP_BATCH_SIZE = HParamS('batch_size', hp.Discrete([1]))
        self.HP_LEANRING_RATE = HParamS('learning_rate', hp.Discrete([0.00003]))
        self.HP_OPTIMISER = HParamS('optimiser', hp.Discrete(['adam']))
        self.HP_LOSS = HParamS('loss', hp.Discrete(['combined']))
        self.HP_ACTIVATION = HParamS('activation', hp.Discrete(['selu']))
        self.HP_KERNEL_INITIALIZER = HParamS('kernel_initializer', hp.Discrete(['lecun_normal']))
        self.HP_DROPOUT = HParamS('drop_out', hp.Discrete([0.0]))
        self.HP_SA_LAMBDA = HParamS('sa_lamdda', hp.Discrete([75]))
        self.HP_LA_LAMBDA = HParamS('la_lamdda', hp.Discrete([1]))
        
        self.parameter_dict = {}
        self.parameter_dict[self.HP_FLOATING_POINT] = self.HP_FLOATING_POINT.domain.values
        self.parameter_dict[self.HP_XLA] = self.HP_XLA.domain.values
        self.parameter_dict[self.HP_EPOCHS] = self.HP_EPOCHS.domain.values
        self.parameter_dict[self.HP_BATCH_SIZE] = self.HP_BATCH_SIZE.domain.values
        self.parameter_dict[self.HP_LEANRING_RATE] = self.HP_LEANRING_RATE.domain.values
        self.parameter_dict[self.HP_OPTIMISER] = self.HP_OPTIMISER.domain.values
        self.parameter_dict[self.HP_LOSS] = self.HP_LOSS.domain.values
        self.parameter_dict[self.HP_ACTIVATION] = self.HP_ACTIVATION.domain.values
        self.parameter_dict[self.HP_KERNEL_INITIALIZER] = self.HP_KERNEL_INITIALIZER.domain.values
        self.parameter_dict[self.HP_DROPOUT] = self.HP_DROPOUT.domain.values
        self.parameter_dict[self.HP_SA_LAMBDA] = self.HP_SA_LAMBDA.domain.values
        self.parameter_dict[self.HP_LA_LAMBDA] = self.HP_LA_LAMBDA.domain.values
        
        
        if search_type == 'grid':
            self.parameter_space = ParameterGrid(self.parameter_dict)
        else:
            raise ValueError('Invalid \'search_type\' input. Given: {}'.format(search_type))
        
        
    def __iter__(self):
        parameter_list = list(self.parameter_space)
        for parameter in parameter_list:
            yield parameter


if __name__ == '__main__':
    config = HyperParameters(search_type='grid')
    for i in config:
        print(i)