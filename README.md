# machine_learning_trading_bot
Multiple supervised machine learning classifiers are used and tested to enhance trading signals' accuracy and trading bot's ability to adapt to new data.



conclusions about the performance of the baseline trading algorithm:
Based on the testing report and the SVM cumulative return plot, the SVM model performed well from the beginning of the period until mid 2018. That’s when the actual and predicted returns start to be slightly differ. And the difference kept growing until the beginning of 2020, then it again drifted apart. The testing report indicates the model predicts well for buy signals with .96 recall rate, while it performed poorly predicting sell signals (.04 recall).
The SVM model made trading decisions that outperformed the actual returns in the scond half of the market data according to the plot. Overall, dispite the volatility, the SVM model's trading strategy produced a higher cumulative return value than the original.

###Tune the Baseline Trading Algorithm
####Tune the training algorithm by adjusting the size of the training dataset.
         training_begin             training_end             testing_begin             testing_end              DatePffset(month)             accuracy
Orig.    2015-04-02                 2015-07-02               2015-07-06                2021-01-22                3                            0.55
Incr.    2015-04-02                 2016-04-02               2016-04-04                2021-01-22                20                           0.57
Decr.    2015-04-02                 2015-05-02               2015-05-04                2021-01-22                1                            0.55

>What impact resulted from increasing or decreasing the training window?
By increasing testing dataset to 20 months, the prediction accuracy has improved to 0.57, especially the recall score for buying signal has increased to 1. Also the stragegy returns greatly out performed actual returns from the plot. By decreasing the training dataset to 1 month, the accuracy score has decreased. And the cumulative return brought by two strategies shows greater differences.

####Tune the trading algorithm by adjusting the SMA input features.
          
          short_window      long_window       accuracy        cumulative_return
Orig.     4                 100               0.55            outperform
Shor.     30                100               0.56            same
Long.     4                 200               0.45            greatly under perform
Both      7                 90                0.56            slightly under perform

>What impact resulted from increasing or decreasing either or both of the SMA windows?
As shown in the table above: when increasing the short window, the accuracy score increased slightly but the cumulative return shows the same. When increase the long window, the accuracy score decreased badly to 0.45 and the cumulative return delivered by bot was greatly under perform comparing to the original trading strategy. And lastly, when change both windows, the accuracy score appeared a subtle increase and the bot is slightly under perform to the original strategy.

###Choose the set of parameters that best improved the trading algorithm returns
It shows that the current values of windows produce the highest cumulative return, yet by increasing the training dataset to 20 months, the machine learning algorith has delivered a cumulative return as high as 1.8, which is higher than modifying other parameters.


###Evaluate a new machine learning classifier
In this section, you’ll use the original parameters that the starter code provided. But, you’ll apply them to the performance of a second machine learning model.
Import a new classifier, such as AdaBoost, DecisionTreeClassifier, or LogisticRegression
Did this new model perform better or worse than the provided baseline model? 
Did this new model perform better or worse than your tuned trading algorithm?


Create an Evaluation Report
In the previous sections, you updated your README.md file with your conclusions. To accomplish this section, you need to add a summary evaluation report at the end of the README.md file. For this report, express your final conclusions and analysis. Support your findings by using the PNG images that you created.
