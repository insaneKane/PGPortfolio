from __future__ import absolute_import, division, print_function
import numpy as np
from pgportfolio.trade import trader
from pgportfolio.marketdata.datamatrices import DataMatrices
import logging
from pgportfolio.tools.trade import calculate_pv_after_commission
from datetime import datetime,time

class BackTest(trader.Trader):
    def __init__(self, config, net_dir=None, agent=None, agent_type="nn"):
        trader.Trader.__init__(self, 300, config, 0, net_dir,
                               initial_BTC=1, agent=agent, agent_type=agent_type)
        self.config = config #My Addition can cause error TODO
        
        if agent_type == "nn":
            data_matrices = self._rolling_trainer.data_matrices
        elif agent_type == "traditional":
            config["input"]["feature_number"] = 1
            data_matrices = DataMatrices.create_from_config(config)
        else:
            raise ValueError()
        self.__test_set = data_matrices.get_test_set()
        self.__test_length = self.__test_set["X"].shape[0]
        self._total_steps = self.__test_length
        self.__test_pv = 1.0
        self.__test_pc_vector = []
        self.__period = config['input']['global_period']
    @property
    def test_pv(self):
        return self.__test_pv

    @property
    def test_pc_vector(self):
        return np.array(self.__test_pc_vector, dtype=np.float32)

    def finish_trading(self):
        self.__test_pv = self._total_capital

        """
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(self._rolling_trainer.data_matrices.sample_count)),
               self._rolling_trainer.data_matrices.sample_count)
        fig.tight_layout()
        plt.show()
        """

    def _log_trading_info(self, time, omega):
        pass

    def _initialize_data_base(self):
        pass

    def _write_into_database(self):
        pass

    def __get_matrix_X(self):
        return self.__test_set["X"][self._steps]

    def __get_matrix_y(self):
        return self.__test_set["y"][self._steps, 0, :]

    def __get_matrix_last_X(self):
        return self._rolling_trainer.last_info["X"]

    def __get_matrix_last_y(self):
        return self._rolling_trainer.last_info["y"][0, :]
        

    def rolling_train(self, online_sample=None):
        self._rolling_trainer.rolling_train()

    def online_rolling_train(self, batch_data):
        self._rolling_trainer.online_rolling_train(batch_data)

    '''Update Rolling Trainer with New Data'''
    def update_matrix(self):
        #self._rolling_trainer.update_data()
        data_matrices = self._rolling_trainer.data_matrices
        self.__test_set = data_matrices.get_test_set()
        self.__test_length = self.__test_set["X"].shape[0]
       
    def generate_history_matrix(self):
        time = self._rolling_trainer.data_matrices.get_current_time(self._steps)
        inputs = self.__get_matrix_X()
        if self._agent_type == "traditional":
            inputs = np.concatenate([np.ones([1, 1, inputs.shape[2]]), inputs], axis=1)
            inputs = inputs[:, :, 1:] / inputs[:, :, :-1]
        return inputs

    def generate_realtime_history_matrix(self):
        inputs = self.__get_matrix_last_X()
        return inputs

    def get_last_output(self):
        return self._rolling_trainer.last_info["y"]

    def trade_by_strategy(self, omega):
        logging.info("the step is {}".format(self._steps))
        logging.debug("the raw omega is {}".format(omega))
        outputs = self.__get_matrix_y()
        future_price = np.concatenate((np.ones(1), self.__get_matrix_y()))
        logging.info("future_price : {}".format(future_price))
        pv_after_commission = calculate_pv_after_commission(omega, self._last_omega, self._commission_rate)
        portfolio_change = pv_after_commission * np.dot(omega, future_price)
        self._total_capital *= portfolio_change
        self._last_omega = pv_after_commission * omega * \
                           future_price /\
                           portfolio_change
        logging.debug("the portfolio change this period is : {}".format(portfolio_change))
        self.__test_pc_vector.append(portfolio_change)
    
    def trade_by_online_strategy(self, omega):
        logging.debug("LAST OMEGA IS {}".format(omega))
        real_output = self.__get_matrix_last_y()
        #print("Last Value : {} Shape : {}".format(real_output, real_output.shape))
        future_price = np.concatenate((np.ones(1), real_output))
        logging.info("FUTURE PRICE : {}".format(future_price))
        pv_after_commission = calculate_pv_after_commission(omega, self._last_omega, self._commission_rate)
        portfolio_change = pv_after_commission * np.dot(omega, future_price)
        self._total_capital *= portfolio_change
        self._last_omega = pv_after_commission * omega * \
                           future_price /\
                           portfolio_change
        logging.debug("TOTAL PORTFOLIO CHANGE: {}".format(portfolio_change))
        self.__test_pc_vector.append(portfolio_change)
