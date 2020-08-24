# 📈 How to use deep reinforcement learning to automate stock trading

## 💡  Original intention

Recently, due to the impact of the new crown epidemic, the stock market has fallen one after another. As a small cabbage and a small leek, it has a bold idea of ​​bargaining and took out the little remaining private money.

The next day, it plummeted, I increased the position

On the third day, it fell again and I increased the position

On the third day, it fell again, and I increased the position...


<img src="img/2020-03-27-10-45-59.png" alt="drawing" width="50%"/>

After some wrong operations, the results were horrible. The first time I bought stocks, I was beaten by the stock market for a while, and was mercilessly mocked by my wife. After learning from the pain, I decided to change my mind: **How to use deep reinforcement learning to automatically simulate stock trading?** Experiment to verify whether you can get revenue.

## 📖 The difference between supervised learning and reinforcement learning

Supervised learning (such as LSTM) can predict the price of future stocks based on various historical data, determine whether the stock is rising or falling, and help people make decisions.

<img src="img/2020-03-25-18-55-13.png" alt="drawing" width="50%"/>

Reinforcement learning is another branch of machine learning that takes appropriate actions (Action) to maximize the final reward when making decisions. Different from supervised learning to predict the future value, reinforcement learning outputs a series of actions (for example: buy, hold, sell) according to the state of the input (such as the opening price and closing price of the day), so as to maximize the final profit and realize Automatic trading.

<img src="img/2020-03-25-18-19-03.png" alt="drawing" width="50%"/>

## 🤖 OpenAI Gym stock trading environment

### Observation

The strategy network observes various parameters of a stock, such as opening price, closing price, and transaction volume. Part of the value will be a very large value, such as transaction amount or transaction volume, which may be millions, tens of millions or even larger. In order to converge the network during training, the observed state data must be normalized and transformed to `[-1, 1]` Within the interval.

|parameter name|Parameter|Description	Description|
|---|---|---|
|date|Exchange market date|Format：YYYY-MM-DD|
|code|Securities code|Format：sh.600000。sh：Shanghai，sz：Shenzhen|
|open|Opening price today|Precision: 4 digits after the decimal point; Unit: RMB|
|high|Highest price	|Precision: 4 digits after the decimal point; Unit: RMB|
|low|Lowest price	|Precision: 4 digits after the decimal point; Unit: RMB|
|close|Closing price today|Precision: 4 digits after the decimal point; Unit: RMB|
|preclose|Yesterday's closing price	|Precision: 4 digits after the decimal point; Unit: RMB|
|volume|The number of transactions	|Unit: share|
|amount|Turnover|Precision: 4 digits after the decimal point; Unit: RMB|
|adjustflag|Restoration status|Non-restoration, pre-restoration, post-restoration|
|turn|Turnover rate|Precision: 6 digits after the decimal point; unit:%|
|tradestatus|trading status|1: Normal trading 0: Trading suspension|
|pctChg|Change (%)|Precision: 6 digits after the decimal point|
|peTTM|Rolling price-earnings ratio|Precision: 6 digits after the decimal point|
|psTTM|Rolling market sales ratio|Precision: 6 digits after the decimal point|
|pcfNcfTTM|Rolling price to cash rate|Precision: 6 digits after the decimal point|
|pbMRQ|P/B ratio|Precision: 6 digits after the decimal point|

### Action

Assuming that the transaction has three operations: buy , sell and hold , the action ( `action` ) is defined as an array of length 2

- `action[0]` Is the operation type;
- `action[1]` Indicates the percentage of buying or selling;

| Action type `action[0]` | Description |
|---|---|
| 1 | Buy in  `action[1]`|
| 2 | Sell  `action[1]`|
| 3 | maintain |

Note that when the type of action  `action[0] = 3`it means do not buy nor sell stocks, then  `action[1]` the value of no practical significance, the network in the training process, Agent will slowly learn this information.

### Reward

The design of the reward function is crucial to the goal of reinforcement learning. In the context of stock trading, the most important thing to care about is the current profit, so the current profit is used as the reward function. That is `Current principal + stock value-initial principal = profit` 

```python
# profits
reward = self.net_worth - INITIAL_ACCOUNT_BALANCE
reward = 1 if reward > 0 else reward = -100
```

In order to make the network learn the profit strategy faster, when the profit is negative, give the network a larger penalty (`-100`)。

### Strategy gradient

Because the value of the action output is continuous, an optimization algorithm based on policy gradients is used. Among them, the [PPO](https://arxiv.org/abs/1707.06347) algorithm is more well-known . OpenAI and many documents have regarded PPO as the preferred algorithm in reinforcement learning research. PPO optimization algorithm Python implementation refers to [stable-baselines](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html) .

## 🕵️‍ Simulation experiment

### Environmental installation

```sh
# Virtual environment
virtualenv -p python3.6 venv
source ./venv/bin/activate
# install the library dependencies 
pip install -r requirements.txt
```

### Stock data acquisition

The stock securities data set comes from [baostock](http://baostock.com/baostock/index.php/%E9%A6%96%E9%A1%B5) , a free and open source securities data platform that provides Python APIs.

```bash
>> pip install baostock -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn
```

Data acquisition code refer to  [get_stock_data.py](https://github.com/wangshub/RL-Stock/blob/master/get_data.py)

```python
>> python get_stock_data.py
```

The stock data of the past 20 years is divided into a training set, and the last month's data is used as a test set to verify the effectiveness of the reinforcement learning strategy. Divided as follows

| `1990-01-01` ~ `2019-11-29` | `2019-12-01` ~ `2019-12-31` |
|---|---|
| Training set | Test set |

### Validation results

**Single stock**

- Initial principal `10000`
- Stock code: `sh.600036`(China Merchants Bank - 招商银行)
- Training set: `stockdata/train/sh.600036.招商银行.csv`
- Test set:  `stockdata/test/sh.600036.招商银行.csv`
- Simulation operation `20` day, a final profit of about `400`

<img src="img/sh.600036.png" alt="drawing" width="70%"/>

**Multiple stocks**

Select `1002` stocks, training, total

- Profit： `44.5%`
- No loss or no profit:  `46.5%`
- Loss：`9.0%`

<img src="img/pie.png" alt="drawing" width="50%"/>

<img src="img/hist.png" alt="drawing" width="50%"/>

## 👻 At Last

- The stock Gym environment mainly refers to [Stock-Trading-Environment](https://github.com/notadamking/Stock-Trading-Environment) , and the observation state, reward function and training set are modified.


## 📚 References

- Y. Deng, F. Bao, Y. Kong, Z. Ren and Q. Dai, "Deep Direct Reinforcement Learning for Financial Signal Representation and Trading," in IEEE Transactions on Neural Networks and Learning Systems, vol. 28, no. 3, pp. 653-664, March 2017.
- [Yuqin Dai, Chris Wang, Iris Wang, Yilun Xu, "Reinforcement Learning for FX trading"](http://stanford.edu/class/msande448/2019/Final_reports/gr2.pdf)
- Chien Yi Huang. Financial trading as a game: A deep reinforcement learning approach. arXiv preprint arXiv:1807.02787, 2018.
- [Create custom gym environments from scratch — A stock market example](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e)
- [notadamking/Stock-Trading-Environment](https://github.com/notadamking/Stock-Trading-Environment)
- [Welcome to Stable Baselines docs! - RL Baselines Made Easy](https://stable-baselines.readthedocs.io/en/master)
