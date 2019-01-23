# Phasor-Transform-for-Automatic-Delineation-of-Single-lead-ECG-Fiducial-Points

Custom implementation of signal processing based approach For ECG signal fiducial points delineation as presented in Martinez et. al 2010 paper (http://iopscience.iop.org/article/10.1088/0967-3334/31/11/005/meta)

Accurate fiducial point detection in ECG signals is a challenging problem given inherent variability of ECG signals, especially while working with high sampling frequency signals. Martinez 2010 paper presentes a Phasor transform based algorithm to detect these waves. This is a custom implementation of the algorithm on private dataset available that has high resolution with 1000 sampling frequency and very challenging data samples for patients with heart conditions.

Relatively good performance received on this dataset. Performance reported in the paper is on signals with 360 and 128 sampling frequency that are relatively simple. However, in Normative conditions, this approch performed pretty well, not so in signnals from patients with known heart conditions. Although strict signal processing based approches with threholds as in this paper typically do not generalize well, they can be used if we cannot afford the cost of well generalizable machine learning models.

**A few more result images are in sample_results directory**  
<p align="center">
<img src="https://github.com/jaideepmurkute/Phasor-Transform-for-Automatic-Delineation-of-Single-lead-ECG-Fiducial-Points/blob/master/sample_results/result_1.png" width="500" height="400" align="center">
</p>

**Performance Summary on Normative signals:**

**Mean and standard deviations of fiducial points detected: q_onset, r_peak, s_end, t_peak, t_end.**  
mean qrs_on_errors:  10.021447721179625  
sd qrs_on_errors:  12.220914304157956  


mean qrs_off_errors:  -6.7560321715817695  
sd qrs_off_errors:  34.764423547328576  


mean t_peak_errors:  2.9705093833780163  
sd t_peak_errors:  43.500598609274256  


mean t_off_errors:  16.386058981233244  
sd t_off_errors:  46.85561667175976  


qrs_sens:  0.9690527857077808 #Sensitivity of QRS wave detection  
qrs_pp:  0.84611016864457 #Positive Predictivity of QRS wave detection  


t_sens:  0.7490091340932458 #Sensitivity of T wave detection  
t_pp:  0.8990215369063009 #Positive Predictivity of T wave detection  


***Accuracy of QTc interval**  

mean QTc_error:  2.213326260496839  
sd QTc error:  12.816471937185646  

**Dependencies**  
Python : 3.6.5  
sklearn : 0.20.0  
numpy : 1.15.3  
matplotlib : 3.0.1  
