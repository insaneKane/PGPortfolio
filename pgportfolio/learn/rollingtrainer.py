from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pgportfolio.learn.tradertrainer import TraderTrainer
import logging
import tflearn
import numpy as np

class RollingTrainer(TraderTrainer):
    def __init__(self, config, restore_dir=None, save_path=None, agent=None, device="cpu"):
        config["training"]["buffer_biased"] = config["trading"]["buffer_biased"]
        config["training"]["learning_rate"] = config["trading"]["learning_rate"]
        TraderTrainer.__init__(self, config, restore_dir=restore_dir, save_path=save_path,
                               agent=agent, device=device)

    @property
    def agent(self):
        return self._agent

    @property
    def coin_list(self):
        return self._matrix.coin_list

    @property
    def data_matrices(self):
        return self._matrix

    @property
    def rolling_training_steps(self):
        return self.config["trading"]["rolling_training_steps"]

    def update_data(self):
        self.update_matrix()
        print("Update Matrix With new Data")

    def __rolling_logging(self):
        fast_train = self.train_config["fast_train"]
        if not fast_train:
            tflearn.is_training(False, self._agent.session)

            v_pv, v_log_mean = self._evaluate("validation",
                                              self._agent.portfolio_value,
                                              self._agent.log_mean)
            t_pv, t_log_mean = self._evaluate("test", self._agent.portfolio_value, self._agent.log_mean)
            loss_value = self._evaluate("training", self._agent.loss)

            logging.info('training loss is %s\n' % loss_value)
            logging.info('the portfolio value on validation asset is %s\nlog_mean is %s\n' %
                         (v_pv,v_log_mean))
            logging.info('the portfolio value on test asset is %s\n mean is %s' % (t_pv,t_log_mean))

    def decide_by_history(self, history, last_w):
        result = self._agent.decide_by_history(history, last_w)
        return result

    def rolling_train(self, online_w=None):
        steps = self.rolling_training_steps
        if steps > 0:
            self._matrix.append_experience(online_w)
            for i in range(steps):
                x, y, last_w, w = self.next_batch()
                self._agent.train(x, y, last_w, w)
            self.__rolling_logging()

    def online_rolling_train(self, batch_data):
        x = batch_data["inputs"]
        y = batch_data["outputs"] 
        last_w = batch_data["last_weights"]
        w = batch_data["ws"]
        #input("X shape : {} y shape : {} last_w shape : {}".format(len(X), len(y), len(last_w)))
        input("X shape : {} y shape : {} last_w shape : {}".format(x.shape, y.shape, last_w.shape))
        self._agent.train(x, y, last_w, w)
        self.__rolling_logging()
