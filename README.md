Feature Selection:
['xy interval', 'xz interval', 'xa interval', 'xb interval', 'xc interval', 'y amplitude', 'z amplitude', 'a amplitude', 'b amplitude', 'c amplitude']

Prediction's MAD on whole set (Training + Testing) 
Name: 資育-許。婷, MAD SBP: 1.6264, MAD DBP: 2.2996
Name: becca, MAD SBP: 2.7584, MAD DBP: 3.0929
Name: shaleen, MAD SBP: 3.6051, MAD DBP: 1.8717
Name: peggy, MAD SBP: 2.0378, MAD DBP: 1.1286
Name: 資育-蔡。華, MAD SBP: 4.5329, MAD DBP: 3.8542
Name: 資育-黃。穎, MAD SBP: 2.7395, MAD DBP: 2.1111
Name: andy, MAD SBP: 3.4199, MAD DBP: 3.5543
Name: 資育-邱。儒, MAD SBP: 3.8887, MAD DBP: 2.3877
Name: 資育-盧o, MAD SBP: 4.1195, MAD DBP: 1.0482

Training Set MAD:
SBP: 5.8417
DBP: 4.0882
Test Set MAD:
SBP: 6.9962
DBP: 5.5620
Training Set Mean Error:
SBP: -0.0000
DBP: -0.0000
Test Set Mean Error:
SBP: -5.0307
DBP: -3.7455

Regression Formula:
Systolic = 65.1926 + 0.9249 * xy interval - 0.0943 * xz interval - 0.4948 * xa interval + 0.5346 * xb interval - 0.1047 * xc interval + 13.7501 * y amplitude + 38.2074 * z amplitude + 33.2244 * a amplitude + 60.3757 * b amplitude + 197.7610 * c amplitude  (WHY???)
Diastolic = 83.6534 + 0.8271 * xy interval - 0.1657 * xz interval - 0.4103 * xa interval + 0.3057 * xb interval - 0.0668 * xc interval + 5.3643 * y amplitude + 37.1302 * z amplitude + 12.3114 * a amplitude + 31.9776 * b amplitude + 196.3875 * c amplitude