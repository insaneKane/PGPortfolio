from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
from pgportfolio.learn.rollingtrainer import RollingTrainer
import logging
import time


class Trader:
    def __init__(self, waiting_period, config, total_steps, net_dir, agent=None, initial_BTC=1.0, agent_type="nn"):
        """
        @:param agent_type: string, could be nn or traditional
        @:param agent: the traditional agent object, if the agent_type is traditional
        """
        self._steps = 0
        self._total_steps = total_steps
        self._period = waiting_period
        self._agent_type = agent_type
        self.config = config #My Addition TODO can cause error
        if agent_type == "traditional":
            config["input"]["feature_number"] = 1
            config["input"]["norm_method"] = "relative"
            self._norm_method = "relative"
        elif agent_type == "nn":
            self._rolling_trainer = RollingTrainer(config, net_dir, agent=agent)
            self._coin_name_list = self._rolling_trainer.coin_list
            self._norm_method = config["input"]["norm_method"]
            if not agent:
                agent = self._rolling_trainer.agent
        else:
            raise ValueError()
        self._agent = agent

        # the total assets is calculated with BTC
        self._total_capital = initial_BTC
        self._window_size = config["input"]["window_size"]
        self._coin_number = config["input"]["coin_number"]
        self._commission_rate = config["trading"]["trading_consumption"]
        self._fake_ratio = config["input"]["fake_ratio"]
        self._asset_vector = np.zeros(self._coin_number+1)
        self.__period = config['input']['global_period']
        self._last_omega = np.zeros((self._coin_number+1,))
        self._last_omega[0] = 1.0

        if self.__class__.__name__=="BackTest":
            # self._initialize_logging_data_frame(initial_BTC)
            self._logging_data_frame = None
            # self._disk_engine =  sqlite3.connect('./database/back_time_trading_log.db')
            # self._initialize_data_base()
        self._current_error_state = 'S000'
        self._current_error_info = ''

    def _initialize_logging_data_frame(self, initial_BTC):
        logging_dict = {'Total Asset (BTC)': initial_BTC, 'BTC': 1}
        for coin in self._coin_name_list:
            logging_dict[coin] = 0
        self._logging_data_frame = pd.DataFrame(logging_dict, index=pd.to_datetime([time.time()], unit='s'))

    def generate_history_matrix(self):
        """override this method to generate the input of agent
        """
        pass

    def finish_trading(self):
        pass

    # add trading data into the pandas data frame
    def _log_trading_info(self, time, omega):
        time_index = pd.to_datetime([time], unit='s')
        if self._steps > 0:
            logging_dict = {'Total Asset (BTC)': self._total_capital, 'BTC': omega[0, 0]}
            for i in range(len(self._coin_name_list)):
                logging_dict[self._coin_name_list[i]] = omega[0, i + 1]
            new_data_frame = pd.DataFrame(logging_dict, index=time_index)
            self._logging_data_frame = self._logging_data_frame.append(new_data_frame)

    def trade_by_strategy(self, omega):
        """execute the trading to the position, represented by the portfolio vector w
        """
        pass

    def rolling_train(self):
        """
        execute rolling train
        """
        pass

    def __trade_body(self):
        self._current_error_state = 'S000'
        starttime = time.time()
        inputs = self.generate_history_matrix()
        omega = self._agent.decide_by_history(inputs,
                                              self._last_omega.copy())
        logging.info("Last Omega : {}\nNew Omega : {}".format(self._last_omega, omega))
        self.trade_by_strategy(omega)
        if self._agent_type == "nn":
            self.rolling_train()
        if not self.__class__.__name__=="BackTest":
            print("LAST OMEGA CHANGE")
            self._last_omega = omega.copy()
        logging.info('total assets are %3f BTC' % self._total_capital)
        logging.debug("="*30)
        trading_time = time.time() - starttime
        if trading_time < self._period:
            logging.info("sleep for %s seconds" % (self._period - trading_time))
        self._steps += 1
        return self._period - trading_time

    def start_trading(self):
        try:
            if not self.__class__.__name__=="BackTest":
                current = int(time.time())
                wait = self._period - (current%self._period)
                logging.info("sleep for %s seconds" % wait)
                time.sleep(wait+2)

                while self._steps < self._total_steps:
                    sleeptime = self.__trade_body()
                    time.sleep(sleeptime)
            else:
                while self._steps < self._total_steps:
                    #print("step : {}".format(self.steps))
                    self.__trade_body()
        finally:
            if self._agent_type=="nn":
                self._agent.recycle()
            self.finish_trading()

    #TODO My implementation
    def online_trading_with_nn(self):
        flag = True
        update_step_count = 0
        X = []
        y = []
        last_w = []
        w = []
        #sleep_time = self.__period
        while flag:
            #time.sleep(30)
            try:
                self._current_error_state = 'S000'
                starttime = int(time.time())
                inputs = self.generate_realtime_history_matrix()
                omega = self._agent.decide_by_history(inputs, self._last_omega.copy())

                #TODO Sleep, Update, Get New Data and Y
                print("WAIT_UNTIL_UPDATE")
                wait = self.__period - (starttime%self.__period)
                #time.sleep(wait + 10)
                print("UPDATING DATA")
                self._rolling_trainer.update_data()
                self.trade_by_online_strategy(omega)
                X.append(inputs)
                y.append(omega)
                last_w.append(self._last_omega)
                w.append(0)
                #TODO NEED TO SET w somehow
                if self._agent_type == "nn" and update_step_count == self._rolling_trainer.rolling_training_steps: 
                    #batch = np.array([np.array(X), np.array(y), np.array(last_w), np.array(w)])
                    #input("X shape : {} y shape : {} last_w shape : {}".format(len(X), len(y), len(last_w)))
                    batch = {"inputs" : np.array(X), "outputs" : np.array(y), "last_weights" : np.array(last_w), "ws" : np.array(w)}
                    input("X shape : {} y shape : {} last_w shape : {}".format(batch["inputs"].shape, batch["outputs"].shape, batch["last_weights"].shape))
                    self.online_rolling_train(batch)
                    X = []
                    y = []
                    last_w = []
                    w = []
                    update_step_count = 0
                if not self.__class__.__name__=="BackTest":
                    print("Switch is REAL")
                #self._last_omega = omega.copy()
                logging.info('total assets are %3f BTC' % self._total_capital)
                logging.debug("="*30)
                trading_time = time.time() - starttime
                if trading_time < self._period:
                    logging.info("sleep for %s seconds" % (self._period - trading_time))
                #sleep_time = self._period - trading_time
                update_step_count += 1
            except KeyboardInterrupt:
                self._agent.recycle()
                self.finish_trading()
