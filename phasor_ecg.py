import math
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
from scipy.signal import butter
from scipy.signal import filtfilt
from os import listdir
from os.path import isfile, join
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')
import sys

def get_sensitivity_predictivity(all_test_preds, all_test_labels, mode ):
    '''
    Calculate performance metrics to gauge model performance and compare against other well known approaches.  
    sensitivity = TP / (TP + FN)    - probability of model identifying points it was expected to detect.
    positive predictivity = TP / (TP + FP)    -  Probability that samples detected as one of the waves actually belong to that wave
    '''
    all_qrs_sens = all_qrs_pp = all_t_sens = all_t_pp = 0
    all_QTc_error = []
    k = 0
    RR_interval = 60 #60 heart beats per minute
    for curr_pred, curr_label in zip(all_test_preds, all_test_labels):
        qrs_tp = qrs_fn = qrs_fp = t_tp = t_fn = t_fp = 0
        true_qrs_start = pred_qrs_start = true_t_end = pred_t_end = 0
        '''
        Loop through signal labels and predictions
        '''
        for i in range(800):
            if curr_label[i] == 1 and curr_pred[i] == 1:
                qrs_tp = qrs_tp + 1
            if curr_label[i] != 1 and curr_pred[i] == 1:
                qrs_fp = qrs_fp + 1
            if curr_label[i] == 1 and curr_pred[i] != 1:
                qrs_fn = qrs_fn + 1
            
            if curr_label[i] == 2 and curr_pred[i] == 2:
                t_tp = t_tp + 1
            if curr_label[i] != 2 and curr_pred[i] == 2:
                t_fp = t_fp + 1
            if curr_label[i] == 2 and curr_pred[i] != 2:
                t_fn = t_fn + 1
            
            if pred_qrs_start == 0 and curr_pred[i] == 1:
                pred_qrs_start = i
            if true_qrs_start == 0 and curr_label[i] == 1:
                true_qrs_start = i
            if i+1 <=799:
                if pred_t_end == 0 and curr_pred[i] == 2 and curr_pred[i+1] == 0:
                    pred_t_end = i
                if true_t_end == 0 and curr_label[i] == 2 and (curr_label[i+1] == 0 or i+1==800):    
                    true_t_end = i
            
        pred_QT_interval = pred_t_end - pred_qrs_start
        true_QT_interval = true_t_end - true_qrs_start
        
        pred_QTc = pred_QT_interval / math.pow(RR_interval, 1/3)
        true_QTc = true_QT_interval / math.pow(RR_interval, 1/3)
        
        curr_QTc_error = true_QTc - pred_QTc
        all_QTc_error.append(curr_QTc_error)
        
        if (qrs_tp + qrs_fn)!=0:
            curr_qrs_sens = qrs_tp / (qrs_tp + qrs_fn)
        else:
            curr_qrs_sens = 0
        if t_tp + t_fn != 0:
            curr_t_sens = t_tp / (t_tp + t_fn)
        else:
            curr_t_sens = 0
        if qrs_tp + qrs_fp != 0:
            curr_qrs_pp = qrs_tp / (qrs_tp + qrs_fp)
        else:
            curr_qrs_pp = 0
        
        if t_tp + t_fp != 0:
            curr_t_pp = t_tp / (t_tp + t_fp)
        else:
            curr_t_pp = 0
        
        all_qrs_sens = all_qrs_sens +  curr_qrs_sens    
        all_qrs_pp = all_qrs_pp + curr_qrs_pp
        all_t_sens = all_t_sens + curr_t_sens
        all_t_pp = all_t_pp + curr_t_pp
        k = k + 1
    '''
    computing average performance
    '''
    qrs_sens = all_qrs_sens / len(all_test_preds)
    qrs_pp = all_qrs_pp / len(all_test_preds)
    t_sens = all_t_sens / len(all_test_preds)
    t_pp  = all_t_pp / len(all_test_preds)  
    
    return qrs_sens, qrs_pp, t_sens, t_pp, all_QTc_error

def martinez(mypath, mode):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    try:
        os.remove("Martinez_log_ourDB.txt") 
    except OSError:
        pass
    log = open("Martinez_log_ourDB.txt", "w+")
    all_test_preds = []
    qrs_on_errors = []
    qrs_off_errors = []
    t_on_errors = []
    t_off_errors = []
    all_test_labels = []
    
    total_samples = len(onlyfiles)
    print_every = 10
    iter = 0
    last_message_at = 0
    
    for file in onlyfiles:
        percent_complete = int(iter/total_samples*100)
        if percent_complete % 10 == 0 and last_message_at < percent_complete:
            last_message_at = percent_complete
            print(percent_complete," percent complete \r",end="")
            sys.stdout.flush()
        path = mypath+file
        #print("processing filename: ", file)
        with open(path) as fp:
            ecg_data = fp.read().split(',')
        
            for i, item in enumerate(ecg_data):
                ecg_data[i] = ecg_data[i].split('\n')
            ecg_data[0] = ecg_data[0][6:]
            labels = []
            new_ecg_data = []
            new_ecg_data.append(float(ecg_data[0][0]))
            
            for i in range(1,800):
                if ecg_data[i] not in['',' ']:
                    if len(labels) < 800:
                        labels.append(ecg_data[i][0])
                    if len(new_ecg_data) < 800:
                        new_ecg_data.append(float(ecg_data[i][1]))
                    if len(new_ecg_data) == 800 and len(labels) == 799:
                        labels.append(ecg_data[i+1][0])
                if len(labels)==800:
                    break
            ecg_data = new_ecg_data
            
            abs_ecg_data = []
            for i, item in enumerate(ecg_data[:800]):
                abs_ecg_data.append(abs(ecg_data[i]))
            smooth2 = savgol_filter(ecg_data[:800],201,3)
            ecg_data = np.array(ecg_data[:800])
            
            '''
            Mean normalizing data samples
            '''
            mean = np.mean(ecg_data)
            sd = np.std(ecg_data)
            new_data = (ecg_data - min(ecg_data)) / (max(ecg_data) - min(ecg_data))
            
            rv = 0.001
            phasor_wave = []
            normal_phasor_wave = []
            normal_phasor_wave1 = []
            normal_phasor_wave2 = []
            buttered = []
            
            '''
            High-pass filter to remove baseline wander which can cause degraded performance
            '''
            b, a = butter(N=3, Wn=0.5, btype='high', analog=True, output='ba')
            buttered = filtfilt(b, a, ecg_data, method='gust')
            
            '''Normalize to [-1,1] range by diving signal with r_peak magnitude - highest magnitude'''
            buttered_n = buttered / max(buttered)
            
            '''
            Phasor transform on two sets of signals - custom modifications to Martinez algorithm to improve performance
            '''
            for i, item in enumerate(smooth2):
                phasor_wave.append(math.atan(abs(buttered[i])/rv))
                normal_phasor_wave2.append(np.sqrt(rv**2 + phasor_wave[i]**2)  )
                normal_phasor_wave.append(math.atan(buttered_n[i]/rv))
                normal_phasor_wave1.append(np.sqrt(rv**2 + buttered_n[i]**2)  )
                
            '''
            Clipping abnormal spikes generated at the ends of the wave by high-pass filter
            '''
            for k in range(0,5):
                normal_phasor_wave[k] = 0
            for k in range(795,800):
                normal_phasor_wave[k] = 0
            
            #taking max of 10 to 790 locations to avoid exception caused by r_peak which is max of phasor being considered as those end points
            #which have sudden unnecessary high amplitude caused by transform
            '''
            Detecting r-peak as the point with the maximum phase magnitude
            '''
            r_peak_loc = normal_phasor_wave2.index(max(normal_phasor_wave2[10:790])) 
            r_peak_val = max(normal_phasor_wave2)
            
            '''
            Localising QRS wave as the segment within 0.003 phase variation from r_peak phase value.
            '''
            qrs_seg_low = 0
            qrs_seg_high = 0
            for i in reversed(np.arange(r_peak_loc-50, r_peak_loc)):
                if phasor_wave[i] <= phasor_wave[r_peak_loc]-0.003:
                    qrs_seg_low = i
            for i in range(r_peak_loc, r_peak_loc+50):
                if phasor_wave[i] <= phasor_wave[r_peak_loc]-0.003:
                    qrs_seg_high = i
            
            '''
            find gamma_minus and gamma_plus locations as the locations closest to r_peak, for which the phase is pi/2 of the r_peak phase value of pi/2.
            '''  
            gamma_qrs_minus_loc = 0
            for loc in range(r_peak_loc-100, r_peak_loc):
                if normal_phasor_wave2[loc] <= 0.25*normal_phasor_wave2[r_peak_loc]:
                    gamma_qrs_minus_loc = loc
            gamma_qrs_plus_loc = 0
            for loc in range(r_peak_loc, r_peak_loc+100):
                if normal_phasor_wave[loc] <= 0.25*normal_phasor_wave[r_peak_loc]:
                    gamma_qrs_plus_loc = loc
                    break
            
            if gamma_qrs_plus_loc == 0:
                plt.close()
                plt.plot(normal_phasor_wave,label="normal_phasor_wave")
                plt.plot(buttered_n,label="buttered_n")
                plt.show()
                   
            '''^^^^^DETECTING Q WAVE^^^^^^^^^^^^^^^^^^^^^^^^^'''
            rv_back = 0.005
            '''set up window of 35 mili-seconds'''
            win = buttered[gamma_qrs_minus_loc-35:gamma_qrs_minus_loc]
            backward_window = np.array(win)- np.median(win)
            
            '''
            Apply phasor transform to window with less sensitive parameters(to avoid problems caused by noise) to detect Q start point.
            '''
            PT_backward_window = []
            M_backward_window = []
            for loc in range(0,len(backward_window)):
                PT_backward_window.append(math.atan(abs(backward_window[loc])/rv_back))
                M_backward_window.append(np.sqrt(rv_back**2 + backward_window[loc]**2))
            
            '''assuming no strong negative deflection exists'''
            max_phase_variation = max(PT_backward_window)
            max_phase_variation_loc = PT_backward_window.index(max_phase_variation)
            Q_start = -1
            #for loc in range(max_phase_variation_loc, len(backward_window)):
            for loc in range(0, len(backward_window)):
                if PT_backward_window[loc] <= 0.5*max_phase_variation:
                    Q_start = loc
                    break
            if Q_start != -1:
                Q_start = gamma_qrs_minus_loc-35+Q_start
            
            '''if such strong negative deflection exists in the window'''
            if Q_start == -1:
                #print("Q wave search...No negative deflection found, checking second condition...")
                largest_M_loc = 0
                for loc in range(0, max_phase_variation_loc):
                    if M_backward_window[loc] >= M_backward_window[largest_M_loc] and PT_backward_window[loc] >= 0.5*max_phase_variation:
                        largest_M_loc = loc
                Q_start = gamma_qrs_minus_loc - 35 + largest_M_loc
            
            
            '''^^^^^^^^^^^^^^^DETECTING S WAVE^^^^^^^^^^^^^^^^^^^'''
            rv_for = 0.005
            '''set up window of 35 mili-seconds'''
            win = buttered[gamma_qrs_plus_loc:gamma_qrs_plus_loc+35]
            forward_window = np.array(win)- np.median(win)
            
            '''
            Again apply phasor transform on new window to detect s_end location
            '''
            PT_forward_window = []
            M_forward_window = []
            for loc in range(0,len(forward_window)):
                PT_forward_window.append(math.atan(abs(forward_window[loc])/rv_for))  
                M_forward_window.append(np.sqrt(rv_for**2 + forward_window[loc]**2))
                
            '''assuming strong negative deflection exists'''
            max_phase_variation = max(PT_forward_window)
            max_phase_variation_loc = PT_forward_window.index(max_phase_variation)
            S_start = -1
            for loc in range(max_phase_variation_loc, len(backward_window)):
                if PT_backward_window[loc] <= 0.5*max_phase_variation:
                    S_start = loc
            if S_start != -1:
                S_start = gamma_qrs_plus_loc + S_start
            
            '''if such strong negative deflection does not exists in window'''
            if S_start == -1:
                largest_M_loc = 0
                for loc in range(0, max_phase_variation_loc):
                    if M_forward_window[loc] >= M_forward_window[largest_M_loc] and PT_forward_window[loc] >= 0.5*max_phase_variation:
                        largest_M_loc = loc
                S_start = gamma_qrs_plus_loc + largest_M_loc
            
            
            true_q= true_r = true_s = true_t_peak = true_t_end = -1
            for i, loc in enumerate(labels[:800]):
                if loc == 's_end':
                    true_s = i
                elif loc == 'r_peak':
                    true_r = i
                elif loc == 'q_onset':
                    true_q = i
                elif loc == 't_peak':
                    true_t_peak = i
                elif loc == 't_end':
                    true_t_end = i
            if true_t_peak == -1:
                continue    
            
            '''^^^^^^^^^^^^^^^^^ DETECTING T WAVE^^^^^^^^^^^^^^^^^^^'''
            '''
            Consider signal after s_end location
            '''
            if S_start+300 <= 799:
                T_window = buttered_n[S_start:S_start+300]
            else:
                T_window = buttered_n[S_start:]
                
            
            T_window_median = np.median(T_window)
            T_window = T_window - T_window_median
            rv_t_wave = 0.003
            PT_T_window = []
            M_T_window = []
            for loc in range(0,len(T_window)):
                PT_T_window.append(math.atan(T_window[loc]/rv_t_wave))  
                M_T_window.append(np.sqrt(rv_t_wave**2 + T_window[loc]**2))
            
            cond_1 = False
            cond_2 = False
            if len(PT_T_window) == 0:
                plt.plot(buttered_n,label="buttered_n")
                plt.plot()
                plt.legend()
                plt.show()
            
            '''Detect t_peak'''
            t_peak_loc = PT_T_window.index(max(PT_T_window))
            for i in range(t_peak_loc,len(PT_T_window)):
                if PT_T_window[i] <= PT_T_window[t_peak_loc] - M_T_window[t_peak_loc]:
                    cond_1 = True
            
            for i in range(0, t_peak_loc):
                if PT_T_window[i] <= PT_T_window[t_peak_loc] - M_T_window[t_peak_loc]:
                    cond_2 = True
            
            
            '''
            Slice out forward window and do median subtraction for normalizing effect
            '''
            forward_window = buttered_n[S_start+t_peak_loc : S_start + t_peak_loc + 65]
            forward_window = forward_window - np.median(forward_window)
            
            '''
            PT on window with much higher rv value - much less sensitive. Since T-waves have pretty low amplitude and inherent noise
            can cause issues to detect landmark points, is sensitive transformation is performed.
            '''
            rv_t_end = 0.005
            new_PT_T_window = []
            new_M_T_window = []
            for loc in range(0,len(forward_window)):
                new_PT_T_window.append(math.atan(forward_window[loc]/rv_t_end))  
                new_M_T_window.append(np.sqrt(rv_t_end**2 + forward_window[loc]**2))
            
            '''
            Compute first derivative of the phasor transformed forward window
            derivative(tan^-1 = 1/(1+x**2))
            '''
            x = np.array(forward_window)/rv_t_end
            derivative_window = 1 /  (x**2 + 1)
            
            simple_derivative_window = []
            '''
            FInd the location with maximum derivative of phasor signal
            '''
            for item in derivative_window:
                simple_derivative_window.append(item)
            max_der_loc = simple_derivative_window.index(max(simple_derivative_window))
            max_der_t_end = new_PT_T_window.index(min(new_PT_T_window[max_der_loc:]))
            
            plt.plot()
            t_end_loc = S_start+t_peak_loc+max_der_t_end
            
            
            '''
            Plot the results.
            figsize is (width,height)
            '''
            fig = plt.figure(figsize=(8,7))
            plt.subplot(2,1,1)
            plt.plot(ecg_data,label='original signal')
            plt.plot(Q_start, ecg_data[Q_start],'bo',color='green',label='Q_start')
            plt.plot(r_peak_loc, ecg_data[r_peak_loc],'bo',color='orange', label='r_peak')
            plt.plot(S_start, ecg_data[S_start],'bo',color='blue',label='s_end')
            plt.plot(S_start+t_peak_loc, ecg_data[S_start+t_peak_loc],'bo',color='black',label='t_peak')
            plt.plot(t_end_loc, ecg_data[t_end_loc],'bo',color='pink',label='t_end')
            plt.title("Predictions")
            plt.legend()
            
            if true_t_end >= 800:
                true_t_end = 799
            plt.subplot(2,1,2)
            plt.plot(ecg_data,label='original signal')
            plt.plot(true_q, ecg_data[true_q],'bo',color='green',label='Q_start')
            plt.plot(true_r, ecg_data[true_r],'bo',color='orange', label='r_peak')
            plt.plot(true_s, ecg_data[true_s],'bo',color='blue',label='s_end')
            plt.plot(true_t_peak, ecg_data[true_t_peak],'bo',color='black',label='t_peak')
            plt.plot(true_t_end, ecg_data[true_t_end],'bo',color='pink',label='t_end')
            plt.title("Ground Truth")
            plt.legend()
            
            plt.tight_layout()
            if mode == 'Normative':
                plt.savefig("Martinez_results_ourDB\\Normative\\"+file.split('.')[0]+"_Martinez.png")
            elif mode == 'HF':
                plt.savefig("Martinez_results_ourDB\\HF\\"+file.split('.')[0]+file.split('.')[1]+"_Martinez.png")
                
            plt.close()
            
            '''generate segment'''
            curr_pred = np.zeros((800))
            t_peak_loc = S_start+t_peak_loc
            
            for i in range(0, 800):
                if i >= Q_start and i <=S_start:
                    curr_pred[i] = 1
                if i > t_peak_loc and i <=t_end_loc:
                    curr_pred[i] = 2
            
            all_test_preds.append(curr_pred)
            
            for j, item in enumerate(labels[:800]):
                if item == 'B':
                    labels[j] = 0
            for i in range(true_q, true_s+1):
                labels[i] = 1
            for i in range(true_t_peak, true_t_end+1):
                labels[i] = 2       
            
            all_test_labels.append(labels[:800])
            
            qrs_on_errors.append(true_q - Q_start)
            qrs_off_errors.append(true_s - S_start)
            t_on_errors.append(true_t_peak - t_peak_loc)
            t_off_errors.append(true_t_end - t_end_loc)
            
            iter = iter + 1
    
    print("Plots have been saved to relative location: ", '\\Martinez_results_ourDB\\Normative\\')
        
    qrs_sens, qrs_pp, t_sens, t_pp, all_QTc_error = get_sensitivity_predictivity(all_test_preds, all_test_labels, mode )
    print("mean qrs_on_errors: ",np.mean(qrs_on_errors))
    print("sd qrs_on_errors: ", np.std(qrs_on_errors))
    
    print("mean qrs_off_errors: ",np.mean(qrs_off_errors))
    print("sd qrs_off_errors: ", np.std(qrs_off_errors))
    
    
    print("mean t_peak_errors: ",np.mean(t_on_errors))
    print("sd t_peak_errors: ", np.std(t_on_errors))
    
    print("mean t_off_errors: ",np.mean(t_off_errors))
    print("sd t_off_errors: ", np.std(t_off_errors))
    
    print("qrs_sens: ", qrs_sens)
    print("qrs_pp: ", qrs_pp)
    
    print("t_sens: ", t_sens)
    print("t_pp: ", t_pp)
    
    print("mean QTc_error: ", np.mean(all_QTc_error))
    print("sd QTc error: ", np.std(all_QTc_error))
    
            
if __name__ == '__main__':
    mypath = input("Enter path of datafiles: ")
    
    mode = input("Mode of heart signals 1]Normative  2]Heart-Failure   (If not known Enter Either one): ")
    
    print(" \033[1;32;47m Note: this implementation assumes signals of length 800,\
        a trivial modification to base code should handle arbitrary length inputs.\
        Work for such modifications is under progress.")
    print("Processing...")
    #mypath = 'C:\\Users\\jaide\\eclipse-workspace\\ECG_Segmentation\\ECGDATA\\Truncated\\Normative\\'
    #mode = 'Normative'
    
    #mypath = 'C:\\Users\\jaide\\eclipse-workspace\\ECG_Segmentation\\ECGDATA\\Truncated\\HeartFailure\\'
    #mode = 'HF'
    
    martinez(mypath,mode)