import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import json
from sklearn.utils import shuffle
import os
import time
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import itertools
from sklearn.metrics import  confusion_matrix
from ConfusionMatrix import ConfusionMatrix
from keras_flops import get_flops

class data_cls:
    def __init__(self, train_test, **kwargs):
        col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
                     "dst_bytes", "land_f", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                     "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                     "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                     "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                     "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                     "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                     "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                     "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                     "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "labels", "dificulty"]
        self.index = 0
        # Data formated path and test path.
        self.loaded = False
        self.train_test = train_test
        self.train_path = kwargs.get('train_path', 'datasets/NSL/KDDTrain+.txt')
        self.test_path = kwargs.get('test_path',
                                    'datasets/NSL/KDDTest+.csv')

        self.formated_train_path = kwargs.get('formated_train_path',
                                              "formated_train_adv.data")
        self.formated_test_path = kwargs.get('formated_test_path',
                                             "formated_test_adv.data")

        self.attack_types = ['normal', 'DoS', 'Probe', 'R2L', 'U2R']
        self.attack_names = []
        self.attack_map = {'normal': 'normal',

                           'back': 'DoS',
                           'land': 'DoS',
                           'neptune': 'DoS',
                           'pod': 'DoS',
                           'smurf': 'DoS',
                           'teardrop': 'DoS',
                           'mailbomb': 'DoS',
                           'apache2': 'DoS',
                           'processtable': 'DoS',
                           'udpstorm': 'DoS',

                           'ipsweep': 'Probe',
                           'nmap': 'Probe',
                           'portsweep': 'Probe',
                           'satan': 'Probe',
                           'mscan': 'Probe',
                           'saint': 'Probe',

                           'ftp_write': 'R2L',
                           'guess_passwd': 'R2L',
                           'imap': 'R2L',
                           'multihop': 'R2L',
                           'phf': 'R2L',
                           'spy': 'R2L',
                           'warezclient': 'R2L',
                           'warezmaster': 'R2L',
                           'sendmail': 'R2L',
                           'named': 'R2L',
                           'snmpgetattack': 'R2L',
                           'snmpguess': 'R2L',
                           'xlock': 'R2L',
                           'xsnoop': 'R2L',
                           'worm': 'R2L',

                           'buffer_overflow': 'U2R',
                           'loadmodule': 'U2R',
                           'perl': 'U2R',
                           'rootkit': 'U2R',
                           'httptunnel': 'U2R',
                           'ps': 'U2R',
                           'sqlattack': 'U2R',
                           'xterm': 'U2R'
                           }
        self.all_attack_names = list(self.attack_map.keys())

        formated = False

        # Test formated data exists
        if os.path.exists(self.formated_train_path) and os.path.exists(self.formated_test_path):
            formated = True

        self.formated_dir = "../datasets/formated/"
        if not os.path.exists(self.formated_dir):
            os.makedirs(self.formated_dir)

        # If it does not exist, it's needed to format the data
        if not formated:
            ''' Formating the dataset for ready-2-use data'''
            self.df = pd.read_csv(self.train_path, sep=',', names=col_names, index_col=False)
            if 'dificulty' in self.df.columns:
                self.df.drop('dificulty', axis=1, inplace=True)  # in case of difficulty

            data2 = pd.read_csv(self.test_path, sep=',', names=col_names, index_col=False)
            if 'dificulty' in data2:
                del (data2['dificulty'])
            train_indx = self.df.shape[0]
            frames = [self.df, data2]
            self.df = pd.concat(frames)

            # Dataframe processing
            self.df = pd.concat([self.df.drop('protocol_type', axis=1), pd.get_dummies(self.df['protocol_type'])],
                                axis=1)
            self.df = pd.concat([self.df.drop('service', axis=1), pd.get_dummies(self.df['service'])], axis=1)
            self.df = pd.concat([self.df.drop('flag', axis=1), pd.get_dummies(self.df['flag'])], axis=1)

            # 1 if ``su root'' command attempted; 0 otherwise
            self.df['su_attempted'] = self.df['su_attempted'].replace(2.0, 0.0)

            # One hot encoding for labels
            self.df = pd.concat([self.df.drop('labels', axis=1),
                                 pd.get_dummies(self.df['labels'])], axis=1)

            # Normalization of the df
            # normalized_df=(df-df.mean())/df.std()
            for indx, dtype in self.df.dtypes.iteritems():
                if dtype == 'float64' or dtype == 'int64':
                    if self.df[indx].max() == 0 and self.df[indx].min() == 0:
                        self.df[indx] = 0
                    else:
                        self.df[indx] = (self.df[indx] - self.df[indx].min()) / (
                                    self.df[indx].max() - self.df[indx].min())

            # Save data
            test_df = self.df.iloc[train_indx:self.df.shape[0]]
            test_df = shuffle(test_df, random_state=np.random.randint(0, 100))
            self.df = self.df[:train_indx]
            self.df = shuffle(self.df, random_state=np.random.randint(0, 100))
            test_df.to_csv(self.formated_test_path, sep=',', index=False)
            self.df.to_csv(self.formated_train_path, sep=',', index=False)

            # Create a list with the existent attacks in the df
            for att in self.attack_map:
                if att in self.df.columns:
                    # Add only if there is exist at least 1
                    if np.sum(self.df[att].values) > 1:
                        self.attack_names.append(att)

    def get_shape(self):
        if self.loaded is False:
            self._load_df()

        self.data_shape = self.df.shape
        # stata + labels
        return self.data_shape

    ''' Get n-rows from loaded data 
        The dataset must be loaded in RAM
    '''

    def get_batch(self, batch_size=100):
        if self.loaded is False:
            self._load_df()

        # Read the df rows
        indexes = list(range(self.index, self.index + batch_size))
        if max(indexes) > self.data_shape[0] - 1:
            dif = max(indexes) - self.data_shape[0]
            indexes[len(indexes) - dif - 1:len(indexes)] = list(range(dif + 1))
            self.index = batch_size - dif
            batch = self.df.iloc[indexes]
        else:
            batch = self.df.iloc[indexes]
            self.index += batch_size

        labels = batch[self.attack_names]

        batch = batch.drop(self.all_attack_names, axis=1)

        return batch, labels

    def get_full(self):
        if self.loaded is False:
            self._load_df()

        labels = self.df[self.attack_names]

        batch = self.df.drop(self.all_attack_names, axis=1)

        return batch, labels

    def _load_df(self):
        if self.train_test == 'train':
            self.df = pd.read_csv(self.formated_train_path, sep=',')  # Read again the csv
        else:
            self.df = pd.read_csv(self.formated_test_path, sep=',')
        self.index = np.random.randint(0, self.df.shape[0] - 1, dtype=np.int32)
        self.loaded = True
        # Create a list with the existent attacks in the df
        for att in self.attack_map:
            if att in self.df.columns:
                # Add only if there is exist at least 1
                if np.sum(self.df[att].values) > 1:
                    self.attack_names.append(att)
        # self.headers = list(self.df)

class QNetwork():
    """
    Q-Network Estimator
    Represents the global model for the table
    """

    def __init__(self, obs_size, num_actions, hidden_size=100,
                 hidden_layers=1, learning_rate=.2, net_name='qv'):
        """
        Initialize the network with the provided shape
        """
        self.net_name = net_name
        self.obs_size = obs_size
        self.num_actions = num_actions

        # Network arquitecture
        self.model = Sequential()
        # Add imput layer
        self.model.add(Dense(hidden_size, input_shape=(obs_size,),
                             activation='relu'))
        # Add hidden layers
        for layers in range(hidden_layers):
            self.model.add(Dense(hidden_size, activation='relu'))
        # Add output layer
        if self.net_name == 'actor':
            self.model.add(Dense(num_actions, activation='softmax'))
        else:
            self.model.add(Dense(num_actions))

        # optimizer = optimizers.SGD(learning_rate)
        # optimizer = optimizers.Adam(alpha=learning_rate)
        optimizer = optimizers.Adam(0.00025)
        # optimizer = optimizers.RMSpropGraves(learning_rate, 0.95, self.momentum, 1e-2)

        # Compilation of the model with optimizer and loss
        if self.net_name == 'actor':
            self.model.compile(loss=sac_loss, optimizer=optimizer)
        else:
            self.model.compile(loss=tf.keras.losses.mse, optimizer=optimizer)

    def predict(self, state, batch_size=1):
        """
        Predicts action values.
        """
        return self.model.predict(state, batch_size=batch_size)

    def update(self, states, q):
        """
        Updates the estimator with the targets.

        Args:
          states: Target states
          q: Estimated values

        Returns:
          The calculated loss on the batch.
        """
        loss = self.model.train_on_batch(states, q)
        return loss

    def copy_model(model):
        """Returns a copy of a keras model."""
        model.save('tmp_model')
        return tf.keras.models.load_model('tmp_model')

def sac_loss(y_true, y_pred):
    """ y_true 是 Q(*, action_n), y_pred 是 pi(*, action_n) """
    qs = 0.2 * tf.math.xlogy(y_pred, y_pred) - y_pred * y_true
    return tf.reduce_sum(qs, axis=-1)



class ReplayMemory(object):
    """Implements basic replay memory"""

    def __init__(self, observation_size, max_size):
        self.observation_size = observation_size
        self.num_observed = 0
        self.max_size = max_size
        self.samples = {
                 'obs'      : np.zeros(self.max_size * 1 * self.observation_size,
                                       dtype=np.float32).reshape(self.max_size,self.observation_size),
                 'action'   : np.zeros(self.max_size * 1, dtype=np.int16).reshape(self.max_size, 1),
                 'reward'   : np.zeros(self.max_size * 1).reshape(self.max_size, 1),
                 'terminal' : np.zeros(self.max_size * 1, dtype=np.int16).reshape(self.max_size, 1),
               }

    def observe(self, state, action, reward, done):
        index = self.num_observed % self.max_size
        self.samples['obs'][index, :] = state
        self.samples['action'][index, :] = action
        self.samples['reward'][index, :] = reward
        self.samples['terminal'][index, :] = done

        self.num_observed += 1

    def sample_minibatch(self, minibatch_size):
        max_index = min(self.num_observed, self.max_size) - 1
        sampled_indices = np.random.randint(max_index, size=minibatch_size)

        s      = np.asarray(self.samples['obs'][sampled_indices, :], dtype=np.float32)
        s_next = np.asarray(self.samples['obs'][sampled_indices+1, :], dtype=np.float32)

        a      = self.samples['action'][sampled_indices].reshape(minibatch_size)
        r      = self.samples['reward'][sampled_indices].reshape((minibatch_size, 1))
        done   = self.samples['terminal'][sampled_indices].reshape((minibatch_size, 1))

        return (s, a, r, s_next, done)


'''
Reinforcement learning Agent definition
'''


class Agent(object):

    def __init__(self, actions, obs_size, **kwargs):
        self.actions = actions
        self.ddqn_time = 100
        self.ddqn_update = self.ddqn_time
        self.num_actions = len(actions)
        self.obs_size = obs_size
        self.alpha = 0.2
        self.net_learning_rate = 0.1
        self.epsilon = kwargs.get('epsilon', 1)
        self.min_epsilon = kwargs.get('min_epsilon', .1)
        self.gamma = kwargs.get('gamma', .001)
        self.minibatch_size = kwargs.get('minibatch_size', 2)
        self.epoch_length = kwargs.get('epoch_length', 100)
        self.ExpRep = kwargs.get('ExpRep', True)
        if self.ExpRep:
            self.memory = ReplayMemory(self.obs_size, kwargs.get('mem_size', 10))

        self.actor_network = QNetwork(self.obs_size, self.num_actions,
                                      kwargs.get('hidden_size', 100),
                                      kwargs.get('hidden_layers', 1),
                                      kwargs.get('learning_rate', .2),
                                      net_name='actor')
        self.q0_network = QNetwork(self.obs_size, self.num_actions,
                                      kwargs.get('hidden_size', 100),
                                      kwargs.get('hidden_layers', 1),
                                      kwargs.get('learning_rate', .2))
        self.q1_network = QNetwork(self.obs_size, self.num_actions,
                                             kwargs.get('hidden_size', 100),
                                             kwargs.get('hidden_layers', 1),
                                             kwargs.get('learning_rate', .2))
        self.v_network = QNetwork(self.obs_size, 1,
                                             kwargs.get('hidden_size', 100),
                                             kwargs.get('hidden_layers', 1),
                                             kwargs.get('learning_rate', .2))
        self.target_v_network = QNetwork(self.obs_size, 1,
                                  kwargs.get('hidden_size', 100),
                                  kwargs.get('hidden_layers', 1),
                                  kwargs.get('learning_rate', .2))

        self.update_target_net(self.target_v_network.model,
                               self.v_network.model, self.net_learning_rate)

        print("Actor:", get_flops(self.actor_network.model,batch_size=1))
        self.actor_network.model.summary()
        print("q:", get_flops(self.q0_network.model, batch_size=1))
        self.q0_network.model.summary()
        print("v:", get_flops(self.target_v_network.model, batch_size=1))
        self.target_v_network.model.summary()

    def learn(self, states, actions, next_states, rewards, done):
        if self.ExpRep:
            self.memory.observe(states, actions, rewards, done)
        else:
            self.states = states
            self.actions = actions
            self.next_states = next_states
            self.rewards = rewards
            self.done = done

    def update_model(self):
        if self.ExpRep:
            (states, actions, rewards, next_states, done) = self.memory.sample_minibatch(self.minibatch_size)
        else:
            states = self.states
            rewards = self.rewards
            next_states = self.next_states
            actions = self.actions
            done = self.done

        pis = self.actor_network.predict(states)
        q0s = self.q0_network.predict(states)
        q1s = self.q1_network.predict(states)

        # 训练执行者
        loss = self.actor_network.model.train_on_batch(states, q0s)

        # 训练评论者
        q01s = np.minimum(q0s, q1s)
        entropic_q01s = pis * q01s - self.alpha * tf.math.xlogy(pis, pis)
        v_targets = tf.math.reduce_sum(entropic_q01s, axis=-1)
        self.v_network.model.fit(states, v_targets, verbose=0)

        next_vs = self.target_v_network.predict(next_states)
        q_targets = rewards.reshape(q0s[range(self.minibatch_size), actions].shape) + self.gamma * (1. - done.reshape(q0s[range(self.minibatch_size), actions].shape)) * \
                    next_vs[:, 0]
        q0s[range(self.minibatch_size), actions] = q_targets
        q1s[range(self.minibatch_size), actions] = q_targets
        self.q0_network.model.fit(states, q0s, verbose=0)
        self.q1_network.model.fit(states, q1s, verbose=0)

        # 更新目标网络
        # self.ddqn_update -= 1
        # if self.ddqn_update == 0:
        #     self.ddqn_update = self.ddqn_time
        #     #            self.target_model_network.model = QNetwork.copy_model(self.model_network.model)
        self.update_target_net(self.target_v_network.model,
                               self.v_network.model, self.net_learning_rate)
        return loss

    def act(self, state, policy):
        raise NotImplementedError

    def update_target_net(self, target_net, evaluate_net, learning_rate=1.):
        target_weights = target_net.get_weights()
        evaluate_weights = evaluate_net.get_weights()
        average_weights = [(1. - learning_rate) * t + learning_rate * e
                for t, e in zip(target_weights, evaluate_weights)]
        target_net.set_weights(average_weights)


class DefenderAgent(Agent):
    def __init__(self, actions, obs_size, **kwargs):
        super().__init__(actions, obs_size, **kwargs)

    def act(self, states):
        # Get actions under the policy
        probs = self.actor_network.model.predict(states)[0]
        actions = np.random.choice(self.actions, size=1, p=probs)
        return actions


class AttackAgent(Agent):
    def __init__(self, actions, obs_size, **kwargs):
        super().__init__(actions, obs_size, **kwargs)

    def act(self, states):
        # Get actions under the policy
        probs = self.actor_network.model.predict(states)[0]
        actions = np.random.choice(self.actions, size=1, p=probs)
        return actions


'''
Reinforcement learning Enviroment Definition
'''


class RLenv(data_cls):
    def __init__(self, train_test, **kwargs):
        data_cls.__init__(self, train_test, **kwargs)
        data_cls._load_df(self)
        self.data_shape = data_cls.get_shape(self)
        self.batch_size = kwargs.get('batch_size', 1)  # experience replay -> batch = 1
        self.iterations_episode = kwargs.get('iterations_episode', 10)
        if self.batch_size == 'full':
            self.batch_size = int(self.data_shape[0] / iterations_episode)

    '''
    _update_state: function to update the current state
    Returns:
        None
    Modifies the self parameters involved in the state:
        self.state and self.labels
    Also modifies the true labels to get learning knowledge
    '''

    def _update_state(self):
        self.states, self.labels = data_cls.get_batch(self)

        # Update statistics
        self.true_labels += np.sum(self.labels).values

    '''
    Returns:
        + Observation of the enviroment
    '''

    def reset(self):
        # Statistics
        self.def_true_labels = np.zeros(len(self.attack_types), dtype=int)
        self.def_estimated_labels = np.zeros(len(self.attack_types), dtype=int)
        self.att_true_labels = np.zeros(len(self.attack_names), dtype=int)

        self.state_numb = 0

        data_cls._load_df(self)  # Reload and random index
        self.states, self.labels = data_cls.get_batch(self, self.batch_size)

        self.total_reward = 0
        self.steps_in_episode = 0
        return self.states.values

    '''
    Returns:
        State: Next state for the game
        Reward: Actual reward
        done: If the game ends (no end in this case)

    In the adversarial enviroment, it's only needed to return the actual reward
    '''

    def act(self, defender_actions, attack_actions):
        # Clear previous rewards
        self.att_reward = np.zeros(len(attack_actions))
        self.def_reward = np.zeros(len(defender_actions))

        attack = [self.attack_types.index(self.attack_map[self.attack_names[att]]) for att in attack_actions]
        # 设置不同的奖励值
        if attack[0] in [3,4]:
            self.def_reward = (np.asarray(defender_actions) == np.asarray(attack)) * 2
            self.att_reward = (np.asarray(defender_actions) != np.asarray(attack)) * 2
        # # elif attack[0] in [3]:
        # #     self.def_reward = (np.asarray(defender_actions) == np.asarray(attack)) * 2
        # #     self.att_reward = (np.asarray(defender_actions) != np.asarray(attack)) * 2
        else:
            self.def_reward = (np.asarray(defender_actions) == np.asarray(attack)) * 1
            self.att_reward = (np.asarray(defender_actions) != np.asarray(attack)) * 1

        self.def_estimated_labels += np.bincount(defender_actions, minlength=len(self.attack_types))
        # TODO
        # list comprehension

        for act in attack_actions:
            self.def_true_labels[self.attack_types.index(self.attack_map[self.attack_names[act]])] += 1

        # Get new state and new true values
        attack_actions = attacker_agent.act(self.states)
        self.states = env.get_states(attack_actions)

        # Done allways false in this continuous task
        self.done = np.zeros(len(attack_actions), dtype=bool)

        return self.states, self.def_reward, self.att_reward, attack_actions, self.done

    '''
    Provide the actual states for the selected attacker actions
    Parameters:
        self:
        attacker_actions: optimum attacks selected by the attacker
            it can be one of attack_names list and select random of this
    Returns:
        State: Actual state for the selected attacks
    '''

    def get_states(self, attacker_actions):
        first = True
        for attack in attacker_actions:
            if first:
                minibatch = (self.df[self.df[self.attack_names[attack]] == 1].sample(1))
                first = False
            else:
                minibatch = minibatch.append(self.df[self.df[self.attack_names[attack]] == 1].sample(1))

        self.labels = minibatch[self.attack_names]
        minibatch.drop(self.all_attack_names, axis=1, inplace=True)
        self.states = minibatch

        return self.states

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == "__main__":

    kdd_train = "E:/TensorflowProject/data/NSL-KDD/KDDTrain+.csv"
    kdd_test = "E:/TensorflowProject/data/NSL-KDD/KDDTest+.csv"

    formated_train_path = "formated_train_adv.data"
    formated_test_path = "formated_test_adv.data"

    # Train batch
    batch_size = 1
    # batch of memory ExpRep
    minibatch_size = 180
    ExpRep = True

    iterations_episode = 180

    # Initialization of the enviroment
    env = RLenv('train', train_path=kdd_train, test_path=kdd_test,
                formated_train_path=formated_train_path,
                formated_test_path=formated_test_path, batch_size=batch_size,
                iterations_episode=iterations_episode)
    # obs_size = size of the state
    obs_size = env.data_shape[1] - len(env.all_attack_names)

    # num_episodes = int(env.data_shape[0]/(iterations_episode)/10)
    num_episodes = 100

    '''
    Definition for the defensor agent.
    '''
    defender_valid_actions = list(range(len(env.attack_types)))  # only detect type of attack
    defender_num_actions = len(defender_valid_actions)

    def_epsilon = 1  # exploration
    min_epsilon = 0.01  # min value for exploration
    def_gamma = 0.001
    def_decay_rate = 0.99

    def_hidden_size = 700
    def_hidden_layers = 2

    def_learning_rate = .2

    defender_agent = DefenderAgent(defender_valid_actions, obs_size,
                                   epoch_length=iterations_episode,
                                   epsilon=def_epsilon,
                                   min_epsilon=min_epsilon,
                                   decay_rate=def_decay_rate,
                                   gamma=def_gamma,
                                   hidden_size=def_hidden_size,
                                   hidden_layers=def_hidden_layers,
                                   minibatch_size=minibatch_size,
                                   mem_size=1000,
                                   learning_rate=def_learning_rate,
                                   ExpRep=ExpRep)
    # Pretrained defender
    # defender_agent.model_network.model.load_weights("models/type_model.h5")

    '''
    Definition for the attacker agent.
    In this case the exploration is better to be greater
    The correlation sould be greater too so gamma bigger
    '''
    attack_valid_actions = list(range(len(env.attack_names)))
    attack_num_actions = len(attack_valid_actions)

    att_epsilon = 1
    min_epsilon = 0.82  # min value for exploration

    att_gamma = 0.001
    att_decay_rate = 0.99

    att_hidden_layers = 2
    att_hidden_size = 100

    att_learning_rate = 0.2

    attacker_agent = AttackAgent(attack_valid_actions, obs_size,
                                 epoch_length=iterations_episode,
                                 epsilon=att_epsilon,
                                 min_epsilon=min_epsilon,
                                 decay_rate=att_decay_rate,
                                 gamma=att_gamma,
                                 hidden_size=att_hidden_size,
                                 hidden_layers=att_hidden_layers,
                                 minibatch_size=minibatch_size,
                                 mem_size=1000,
                                 learning_rate=att_learning_rate,
                                 ExpRep=ExpRep)

    # Statistics
    att_reward_chain = []
    def_reward_chain = []
    att_loss_chain = []
    def_loss_chain = []
    def_total_reward_chain = []
    att_total_reward_chain = []

    # Print parameters
    print("-------------------------------------------------------------------------------")
    print("Total epoch: {} | Iterations in epoch: {}"
          "| Minibatch from mem size: {} | Total Samples: {}|".format(num_episodes,
                                                                      iterations_episode, minibatch_size,
                                                                      num_episodes * iterations_episode))
    print("-------------------------------------------------------------------------------")
    print("Dataset shape: {}".format(env.data_shape))
    print("-------------------------------------------------------------------------------")
    print("Attacker parameters: Num_actions={} | gamma={} |"
          " epsilon={} | ANN hidden size={} | "
          "ANN hidden layers={}|".format(attack_num_actions,
                                         att_gamma, att_epsilon, att_hidden_size,
                                         att_hidden_layers))
    print("-------------------------------------------------------------------------------")
    print("Defense parameters: Num_actions={} | gamma={} | "
          "epsilon={} | ANN hidden size={} |"
          " ANN hidden layers={}|".format(defender_num_actions,
                                          def_gamma, def_epsilon, def_hidden_size,
                                          def_hidden_layers))
    print("-------------------------------------------------------------------------------")

    # Main loop
    attacks_by_epoch = []
    attack_labels_list = []
    for epoch in range(num_episodes):
        start_time = time.time()
        att_loss = 0.
        def_loss = 0.
        def_total_reward_by_episode = 0
        att_total_reward_by_episode = 0
        # Reset enviromet, actualize the data batch with random state/attacks
        states = env.reset()

        # Get actions for actual states following the policy
        attack_actions = attacker_agent.act(states)
        states = env.get_states(attack_actions)

        done = False

        attacks_list = []
        # Iteration in one episode
        for i_iteration in range(iterations_episode):

            attacks_list.append(attack_actions[0])
            # apply actions, get rewards and new state
            act_time = time.time()
            defender_actions = defender_agent.act(states)
            # Enviroment actuation for this actions
            next_states, def_reward, att_reward, next_attack_actions, done = env.act(defender_actions, attack_actions)
            # If the epoch*batch_size*iterations_episode is largest than the df

            attacker_agent.learn(states, attack_actions, next_states, att_reward, done)
            defender_agent.learn(states, defender_actions, next_states, def_reward, done)

            act_end_time = time.time()

            # Train network, update loss after at least minibatch_learns
            if ExpRep and epoch * iterations_episode + i_iteration >= minibatch_size:
                def_loss += defender_agent.update_model()
                att_loss += attacker_agent.update_model()
            elif not ExpRep:
                def_loss += defender_agent.update_model()
                att_loss += attacker_agent.update_model()

            update_end_time = time.time()

            # Update the state
            states = next_states
            attack_actions = next_attack_actions

            # Update statistics
            def_total_reward_by_episode += np.sum(def_reward, dtype=np.int32)
            att_total_reward_by_episode += np.sum(att_reward, dtype=np.int32)

        attacks_by_epoch.append(attacks_list)
        # Update user view
        def_reward_chain.append(def_total_reward_by_episode)
        att_reward_chain.append(att_total_reward_by_episode)
        def_loss_chain.append(def_loss)
        att_loss_chain.append(att_loss)

        end_time = time.time()
        print("\r\n|Epoch {:03d}/{:03d}| time: {:2.2f}|\r\n"
              "|Def Loss {:4.4f} | Def Reward in ep {:03d}|\r\n"
              "|Att Loss {:4.4f} | Att Reward in ep {:03d}|"
              .format(epoch, num_episodes, (end_time - start_time),
                      def_loss, def_total_reward_by_episode,
                      att_loss, att_total_reward_by_episode))

        print("|Def Estimated: {}| Att Labels: {}".format(env.def_estimated_labels,
                                                          env.def_true_labels))
        attack_labels_list.append(env.def_true_labels)

    if not os.path.exists('models'):
        os.makedirs('models')
    # Save trained model weights and architecture, used in test
    defender_agent.actor_network.model.save_weights("models/defender_agent_model_0_1.h5", overwrite=True)
    with open("models/defender_agent_model_0_1.json", "w") as outfile:
        json.dump(defender_agent.actor_network.model.to_json(), outfile)

    formated_test_path = "formated_test_adv.data"

    with open("models/defender_agent_model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("models/opt-8415/defender_agent_model.h5")

    model.compile(loss=sac_loss, optimizer="sgd")

    # Define environment, game, make sure the batch_size is the same in train
    env_test = RLenv('test', formated_test_path=formated_test_path)

    total_reward = 0

    true_labels = np.zeros(len(env_test.attack_types), dtype=int)
    estimated_labels = np.zeros(len(env_test.attack_types), dtype=int)
    estimated_correct_labels = np.zeros(len(env_test.attack_types), dtype=int)

    # states , labels = env.get_sequential_batch(test_path,batch_size = env.batch_size)
    states, labels = env_test.get_full()

    start_time = time.time()
    q = model.predict(states)
    actions = np.argmax(q, axis=1)
    end_time = time.time()
    print("predit time:", end_time -start_time)

    maped = []
    for indx, label in labels.iterrows():
        maped.append(env_test.attack_types.index(env_test.attack_map[label.idxmax()]))

    labels, counts = np.unique(maped, return_counts=True)
    true_labels[labels] += counts

    for indx, a in enumerate(actions):
        estimated_labels[a] += 1
        if a == maped[indx]:
            total_reward += 1
            estimated_correct_labels[a] += 1

    action_dummies = pd.get_dummies(actions)
    posible_actions = np.arange(len(env_test.attack_types))
    for non_existing_action in posible_actions:
        if non_existing_action not in action_dummies.columns:
            action_dummies[non_existing_action] = np.uint8(0)
    labels_dummies = pd.get_dummies(maped)

    normal_f1_score = f1_score(labels_dummies[0].values, action_dummies[0].values)
    dos_f1_score = f1_score(labels_dummies[1].values, action_dummies[1].values)
    probe_f1_score = f1_score(labels_dummies[2].values, action_dummies[2].values)
    r2l_f1_score = f1_score(labels_dummies[3].values, action_dummies[3].values)
    u2r_f1_score = f1_score(labels_dummies[4].values, action_dummies[4].values)

    Accuracy = [normal_f1_score, dos_f1_score, probe_f1_score, r2l_f1_score, u2r_f1_score]
    Mismatch = estimated_labels - true_labels

    acc = float(100 * total_reward / len(states))
    print('\r\nTotal reward: {} | Number of samples: {} | Accuracy = {:.2f}%'.format(total_reward,
                                                                                     len(states), acc))
    outputs_df = pd.DataFrame(index=env_test.attack_types, columns=["Estimated", "Correct", "Total", "F1_score"])
    for indx, att in enumerate(env_test.attack_types):
        outputs_df.iloc[indx].Estimated = estimated_labels[indx]
        outputs_df.iloc[indx].Correct = estimated_correct_labels[indx]
        outputs_df.iloc[indx].Total = true_labels[indx]
        outputs_df.iloc[indx].F1_score = Accuracy[indx] * 100
        outputs_df.iloc[indx].Mismatch = abs(Mismatch[indx])

    print(outputs_df)

    aggregated_data_test = np.array(maped)

    print('Performance measures on Test data')
    print('Accuracy =  {:.4f}'.format(accuracy_score(aggregated_data_test, actions)))
    print('F1 =  {:.4f}'.format(f1_score(aggregated_data_test, actions, average='weighted')))
    print('Precision_score =  {:.4f}'.format(precision_score(aggregated_data_test, actions, average='weighted')))
    print('recall_score =  {:.4f}'.format(recall_score(aggregated_data_test, actions, average='weighted')))

    cnf_matrix = confusion_matrix(aggregated_data_test, actions)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=env.attack_types, normalize=True,
                          title='Normalized confusion matrix')
    # plt.savefig('confusion_matrix_adversarial.svg', format='svg', dpi=1000)
    plt.savefig('confusion_matrix_adversarial.png',  dpi=1000)
