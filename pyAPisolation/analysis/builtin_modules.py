"""
Built-in analysis modules for common electrophysiology analyses.

This module contains the legacy spike and subthreshold analysis modules
that wrap existing functionality while providing the modular interface.
"""

import pandas as pd
import numpy as np
import copy
from .base import AnalysisModule


class SpikeAnalysisModule(AnalysisModule):
    """
    Legacy spike analysis module - preserves existing functionality
    """
    
    def __init__(self):
        super().__init__("spike", "Spike Analysis")
        
    def get_ui_elements(self):
        return {
            'dvdt_thres': 'float',
            'start_time': 'float', 
            'end_time': 'float',
            'thres_to_peak_time': 'float',
            'thres_to_peak_height': 'float',
            'min_peak_height': 'float',
            'bstim': 'bool',
            'bessel': 'float',
            'thres_per': 'float',
            'bspikeFind': 'bool',
            'brunningBin': 'bool', 
            'brawData': 'bool'
        }
    
    def parse_ui_params(self, ui_elements):
        dv_cut = float(ui_elements['dvdt_thres'].text())
        lowerlim = float(ui_elements['start_time'].text())
        upperlim = float(ui_elements['end_time'].text())
        tp_cut = float(ui_elements['thres_to_peak_time'].text())/1000
        min_cut = float(ui_elements['thres_to_peak_height'].text())
        min_peak = float(ui_elements['min_peak_height'].text())
        bstim_find = ui_elements['bstim'].isChecked()
        bessel_filter = float(ui_elements['bessel'].text())
        thresh_frac = float(ui_elements['thres_per'].text())

        return {
            'filter': 0, 
            'dv_cutoff': dv_cut, 
            'start': lowerlim, 
            'end': upperlim, 
            'max_interval': tp_cut,
            'min_height': min_cut, 
            'min_peak': min_peak, 
            'stim_find': bstim_find, 
            'bessel_filter': bessel_filter, 
            'thresh_frac': thresh_frac
        }
    
    def run_individual_analysis(self, abf, selected_sweeps, param_dict, 
                               popup=None, show_rejected=False):
        # Import here to avoid circular imports
        from ipfx.feature_extractor import SpikeFeatureExtractor
        from ..featureExtractor import determine_rejected_spikes
        
        if selected_sweeps is None:
            selected_sweeps = abf.sweepList 
            
        temp_param_dict = copy.deepcopy(param_dict)
        
        # Adjust end time if needed
        if temp_param_dict['end'] == 0.0 or temp_param_dict['end'] > abf.sweepX[-1]:
            temp_param_dict['end'] = abf.sweepX[-1]

        spike_extractor = SpikeFeatureExtractor(
            filter=0,  
            dv_cutoff=temp_param_dict['dv_cutoff'],
            max_interval=temp_param_dict['max_interval'], 
            min_height=temp_param_dict['min_height'], 
            min_peak=temp_param_dict['min_peak'], 
            start=temp_param_dict['start'], 
            end=temp_param_dict['end'], 
            thresh_frac=temp_param_dict['thresh_frac']
        )
        
        spike_df = {}
        rejected_spikes = {} if show_rejected else None
        
        for sweep in selected_sweeps:
            abf.setSweep(sweep)
            spike_df[sweep] = spike_extractor.process(abf.sweepX, abf.sweepY, abf.sweepC)
            
            if show_rejected:
                rejected_spikes[sweep] = pd.DataFrame().from_dict(
                    determine_rejected_spikes(spike_extractor, spike_df[sweep], 
                                            abf.sweepY, abf.sweepX, temp_param_dict)).T
            
            if popup:
                popup.setValue(sweep)
        
        return {
            'spike_df': spike_df,
            'rejected_spikes': rejected_spikes,
            'subthres_df': None
        }
    
    def run_batch_analysis(self, folder_path, param_dict, protocol_name):
        # This would call the existing _inner_analysis_loop method
        # For now, we'll raise NotImplementedError and let the legacy code handle it
        raise NotImplementedError("Batch analysis should use legacy _inner_analysis_loop method")
    
    def save_results(self, results, output_dir, output_tag, save_options=None):
        from ..featureExtractor import save_data_frames
        
        if save_options is None:
            save_options = {'spike_find': True, 'running_bin': True, 'raw_data': True}
            
        save_data_frames(results[0], results[1], results[2], output_dir, output_tag,
                        save_options.get('spike_find', True),
                        save_options.get('running_bin', True), 
                        save_options.get('raw_data', True))


class SubthresholdAnalysisModule(AnalysisModule):
    """
    Legacy subthreshold analysis module - preserves existing functionality
    """
    
    def __init__(self):
        super().__init__("subthres", "Subthreshold Analysis")
        
    def get_ui_elements(self):
        return {
            'subthresSweeps': 'string',
            'stimPer': 'float',
            'startCM': 'float',
            'endCM': 'float',
            'besselFilterCM': 'float'
        }
    
    def parse_ui_params(self, ui_elements):
        try:
            subt_sweeps = np.fromstring(ui_elements['subthresSweeps'].text(), 
                                       dtype=int, sep=',')
            if len(subt_sweeps) == 0:
                subt_sweeps = None
        except:
            subt_sweeps = None
            
        try:
            start_sear = float(ui_elements['startCM'].text())
            end_sear = float(ui_elements['endCM'].text())
        except:
            start_sear = None
            end_sear = None
            
        time_after = float(ui_elements['stimPer'].text())
        
        if start_sear == 0:
            start_sear = None
        if end_sear == 0:
            end_sear = None
            
        return {
            'subt_sweeps': subt_sweeps, 
            'time_after': time_after, 
            'start_sear': start_sear, 
            'end_sear': end_sear
        }
    
    def run_individual_analysis(self, abf, selected_sweeps, param_dict, 
                               popup=None, show_rejected=False):
        from ..featureExtractor import analyze_subthres
        
        subthres_df, _ = analyze_subthres(abf, **param_dict)
        
        return {
            'spike_df': None,
            'rejected_spikes': None,
            'subthres_df': subthres_df
        }
    
    def run_batch_analysis(self, folder_path, param_dict, protocol_name):
        # This would call the existing _inner_analysis_loop_subthres method
        raise NotImplementedError("Batch analysis should use legacy _inner_analysis_loop_subthres method")
    
    def save_results(self, results, output_dir, output_tag, save_options=None):
        from ..featureExtractor import save_subthres_data
        
        avg_df, sweepwise_df = results
        save_subthres_data(avg_df, sweepwise_df, output_dir, output_tag)
