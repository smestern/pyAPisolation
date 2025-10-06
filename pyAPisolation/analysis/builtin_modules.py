"""
Built-in analysis modules for common electrophysiology analyses.

This module contains the legacy spike and subthreshold analysis modules
that wrap existing functionality while providing the modular interface.
"""

import pandas as pd
import numpy as np
import copy
from .base import AnalysisModule, AnalysisParameters


class SpikeAnalysisModule(AnalysisModule):
    """
    Legacy spike analysis module - preserves existing functionality
    """
    
    def __init__(self):
        # Initialize with default spike analysis parameters
        spike_params = AnalysisParameters()
        spike_params.set('dv_cutoff', 20.0)
        spike_params.set('max_interval', 0.005)
        spike_params.set('min_height', 2.0)
        spike_params.set('min_peak', -30.0)
        spike_params.set('thresh_frac', 0.05)
        spike_params.set('bessel_filter', 10000.0)
        spike_params.set('stim_find', False)
        
        super().__init__("spike", "Spike Analysis", spike_params)
    
    def get_default_parameters(self) -> AnalysisParameters:
        """Get default parameters for spike analysis"""
        params = AnalysisParameters()
        params.set('dv_cutoff', 20.0)
        params.set('max_interval', 0.005)
        params.set('min_height', 2.0)
        params.set('min_peak', -30.0)
        params.set('thresh_frac', 0.05)
        params.set('bessel_filter', 10000.0)
        params.set('stim_find', False)
        return params
        
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
    
    def analyze(self, data, selected_sweeps, param_dict):
         # Import here to avoid circular imports
        from ipfx.feature_extractor import SpikeFeatureExtractor
        from ..featureExtractor import determine_rejected_spikes
        from ..featureExtractor import process_file as legacy_analyze
        
        # Use both legacy param_dict and new parameters system
        # Prefer param_dict for backward compatibility
        temp_param_dict = copy.deepcopy(param_dict)
        
        # If param_dict is missing values, fall back to parameters
        if 'dv_cutoff' not in temp_param_dict:
            temp_param_dict['dv_cutoff'] = self.get_parameter('dv_cutoff', 20.0)
        if 'max_interval' not in temp_param_dict:
            temp_param_dict['max_interval'] = self.get_parameter('max_interval', 0.005)
        if 'min_height' not in temp_param_dict:
            temp_param_dict['min_height'] = self.get_parameter('min_height', 2.0)
        if 'min_peak' not in temp_param_dict:
            temp_param_dict['min_peak'] = self.get_parameter('min_peak', -30.0)
        if 'thresh_frac' not in temp_param_dict:
            temp_param_dict['thresh_frac'] = self.get_parameter('thresh_frac', 0.05)
        
        # Update start/end from parameters if available
        if 'start' not in temp_param_dict:
            temp_param_dict['start'] = self.get_parameter('start_time', 0.0)
        if 'end' not in temp_param_dict:
            temp_param_dict['end'] = self.get_parameter('end_time', 0.0)
            
        temp_param_dict = copy.deepcopy(param_dict)
        
        #drop param_dict keys that are not used
        protocol = temp_param_dict.pop("protocol_name", None)
        temp_param_dict.pop("param_dict", None)
        res = legacy_analyze(data.filePath, param_dict=temp_param_dict, protocol_name=protocol)

        return {
            'spike_df': res[1],
            'spike_summary': res[0],
            'running_bin': res[2],
            'subthres_df': None
        }


    def run_individual_analysis(self, file, selected_sweeps, param_dict,
                                popup=None, show_rejected=False):
        # Import here to avoid circular imports
        from ipfx.feature_extractor import SpikeFeatureExtractor
        from ..featureExtractor import determine_rejected_spikes
        from ..featureExtractor import process_file as legacy_analyze
        
        # Use both legacy param_dict and new parameters system
        # Prefer param_dict for backward compatibility
        temp_param_dict = copy.deepcopy(param_dict)
        
        # If param_dict is missing values, fall back to parameters
        if 'dv_cutoff' not in temp_param_dict:
            temp_param_dict['dv_cutoff'] = self.get_parameter('dv_cutoff', 20.0)
        if 'max_interval' not in temp_param_dict:
            temp_param_dict['max_interval'] = self.get_parameter('max_interval', 0.005)
        if 'min_height' not in temp_param_dict:
            temp_param_dict['min_height'] = self.get_parameter('min_height', 2.0)
        if 'min_peak' not in temp_param_dict:
            temp_param_dict['min_peak'] = self.get_parameter('min_peak', -30.0)
        if 'thresh_frac' not in temp_param_dict:
            temp_param_dict['thresh_frac'] = self.get_parameter('thresh_frac', 0.05)
        
        # Update start/end from parameters if available
        if 'start' not in temp_param_dict:
            temp_param_dict['start'] = self.get_parameter('start_time', 0.0)
        if 'end' not in temp_param_dict:
            temp_param_dict['end'] = self.get_parameter('end_time', 0.0)
            
        temp_param_dict = copy.deepcopy(param_dict)
        
        #drop param_dict keys that are not used
        protocol = temp_param_dict.pop("protocol_name", None)
        temp_param_dict.pop("param_dict", None)
        res = legacy_analyze(file, param_dict=temp_param_dict, protocol_name=protocol)

        return {
            'spike_df': res[1],
            'spike_summary': res[0],
            'running_bin': res[2],
            'subthres_df': None
        }
    
    
    def save_results(self, results, output_dir, output_tag, save_options=None):
        from ..featureExtractor import save_data_frames
        
        if save_options is None:
            save_options = {'spike_find': True, 'running_bin': True, 'raw_data': True}
            
        save_data_frames(results[0], results[1], results[2], output_dir, output_tag,
                        save_options.get('spike_find', True),
                        save_options.get('running_bin', True), 
                        save_options.get('raw_data', True))

    def get_results(self, results):
        """
        Convert results to standardized format
        """
        spike_df = results['spike_df']
        rejected_spikes = results.get('rejected_spikes', None)
        
        # Convert to DataFrame if needed
        if isinstance(spike_df, dict):
            spike_df = pd.DataFrame.from_dict(spike_df, orient='index')
        
        return {
            'detailed_data': spike_df,
            'summary_data': rejected_spikes
        }

    def _populate_result(self, result, analysis_results, data):
        """
        Populate AnalysisResult with data from analysis_results.
        """
        #in our case, this is specific to spikes
        result.detailed_data = self.analysis_results['spike_df']
        result.summary_data = self.analysis_results.get('spike_summary', None)
        if 'running_bin' in analysis_results and analysis_results['running_bin'] is not None:
            result.additional_data['running_bin'] = analysis_results['running_bin']
        return result



class SubthresholdAnalysisModule(AnalysisModule):
    """
    Legacy subthreshold analysis module - preserves existing functionality
    """
    
    def __init__(self):
        # Initialize with default subthreshold analysis parameters
        subthres_params = AnalysisParameters()
        subthres_params.set('time_after', 100.0)
        subthres_params.set('bessel_filter_cm', 10000.0)
        
        super().__init__("subthres", "Subthreshold Analysis", subthres_params)
    
    def get_default_parameters(self) -> AnalysisParameters:
        """Get default parameters for subthreshold analysis"""
        params = AnalysisParameters()
        params.set('time_after', 100.0)
        params.set('bessel_filter_cm', 10000.0)
        return params
           
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
    
    def run_individual_analysis(self, abf, selected_sweeps, param_dict=None, 
                               popup=None, show_rejected=False):
        from ..featureExtractor import analyze_subthres

        #convert or use the param_dict
        if param_dict is None:
            param_dict = {}

        #fix the keywords
        param_dict['start_sear'] = param_dict.pop('start', None)
        param_dict['end_sear'] = param_dict.pop('end', None)
        param_dict['savfilter'] = param_dict.pop('filter', None)
        
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


class ResistanceLadder(AnalysisModule):
    """
    Computes the membrane resistance by fitting a linear regression to a collection of I-V pairs. Computed from the sweep data
    """
    def __init__(self):
        super().__init__("resistance_ladder", "Resistance Ladder Analysis")
        self.parameters = self.get_default_parameters()

    def get_default_parameters(self) -> AnalysisParameters:
        """Get default parameters for resistance ladder analysis"""
        params = AnalysisParameters()
        params.set('iv_pairs', [])
        params.set('fit_method', 'linear')
        return params


    def analyze(self, celldata, selected_sweeps, param_dict,
                                popup=None, show_rejected=False):
        
        # Placeholder for actual implementation
        from ..patch_subthres import ladder_rm
        

    
    def save_results(self, results, output_dir, output_tag, save_options=None):
        # Placeholder for actual implementation
        pass