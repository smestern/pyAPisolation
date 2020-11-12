import pandas as pd
import numpy as np
import logging
import pyabf
from ipfx.sweep import Sweep,SweepSet
from ipfx.ephys_data_set import EphysDataSet


class ABFDataSet(EphysDataSet):
    def __init__(self, sweep_info=None, abf_file=None, ontology=None, api_sweeps=True, validate_stim=True):
        super(ABFDataSet, self).__init__(ontology, validate_stim)
        self._abf_data = pyabf.ABF(abf_file)


        if sweep_info is None:
            sweep_info = self.extract_sweep_stim_info()

        self.build_sweep_table(sweep_info)

    @property
    def abf_data(self):
        return self._abf_data


    def extract_sweep_stim_info(self):

        logging.debug("Build sweep table")

        sweep_info = []

        def get_finite_or_none(d, key):

            try:
                value = d[key]
            except KeyError:
                return None

            if np.isnan(value):
                return None

            return value

        for index, sweep_map in enumerate(self._abf_data.sweepList):
            self._abf_data.setSweep(sweep_map)
            sweep_record = {}
            sweep_num = index + 1
            sweep_record["sweep_number"] = sweep_num

            #attrs = self.nwb_data.get_sweep_attrs(sweep_num)

            sweep_record["stimulus_units"] = self.get_stimulus_units(sweep_num)

            sweep_record["bridge_balance_mohm"] = np.nan
            sweep_record["leak_pa"] = np.nan
            sweep_record["stimulus_scale_factor"] = np.nan

            sweep_record["stimulus_code"] = self.get_stimulus_code(sweep_num)
            sweep_record["stimulus_code_ext"] = self.get_stimulus_code_ext(sweep_num)

            if self.ontology:
                sweep_record["stimulus_name"] = self.get_stimulus_code(sweep_num)

            sweep_info.append(sweep_record)

        return sweep_info

    def get_stimulus_units(self, sweep_num):

        unit_str = self.abf_data.sweepUnitsC
        return unit_str

    def get_clamp_mode(self, sweep_num):

        #attrs = self.nwb_data.get_sweep_attrs(sweep_num)
        timeSeriesType = self.abf_data.sweepUnitsY

        if "mV" in timeSeriesType:
            clamp_mode = self.CURRENT_CLAMP
        elif "pA" in timeSeriesType:
            clamp_mode = self.VOLTAGE_CLAMP
        else:
            raise ValueError("Unexpected TimeSeries type {}.".format(timeSeriesType))

        return clamp_mode

    def get_stimulus_code(self, sweep_num):

        stim_code_ext = self.abf_data.protocol

        return stim_code_ext

    def get_stimulus_code_ext(self, sweep_num):

        return self.abf_data.protocol

    def get_recording_date(self):
        return self.abf_data.abfDateTime

    def build_sweep_table(self, sweep_info=None):

        if sweep_info:
            self.add_clamp_mode(sweep_info)
            self.sweep_table = pd.DataFrame.from_records(sweep_info)
        else:
            self.sweep_table = pd.DataFrame(columns=self.COLUMN_NAMES)

    def add_clamp_mode(self, sweep_info):
        """
        Check if clamp mode is available and otherwise detect it
        Parameters
        ----------
        sweep_info
        Returns
        -------
        """

        for sweep_record in sweep_info:
            sweep_number = sweep_record["sweep_number"]
            sweep_record[self.CLAMP_MODE] = self.get_clamp_mode(sweep_number)

    def filtered_sweep_table(self,
                             clamp_mode=None,
                             stimuli=None,
                             stimuli_exclude=None,
                             ):

        st = self.sweep_table

        if clamp_mode:
            mask = st[self.CLAMP_MODE] == clamp_mode
            st = st[mask.astype(bool)]

        if stimuli:
            mask = st[self.STIMULUS_CODE].apply(
                self.ontology.stimulus_has_any_tags, args=(stimuli,), tag_type="code")
            st = st[mask.astype(bool)]

        if stimuli_exclude:
            mask = ~st[self.STIMULUS_CODE].apply(
                self.ontology.stimulus_has_any_tags, args=(stimuli_exclude,), tag_type="code")
            st = st[mask.astype(bool)]

        return st

    def get_sweep_number(self, stimuli, clamp_mode=None):

        sweeps = self.filtered_sweep_table(clamp_mode=clamp_mode,
                                           stimuli=stimuli).sort_values(by=self.SWEEP_NUMBER)

        if len(sweeps) > 1:
            logging.warning(
                "Found multiple sweeps for stimulus %s: using largest sweep number" % str(stimuli))

        if len(sweeps) == 0:
            raise IndexError("Cannot find {} sweeps with clamp mode: {} found ".format(stimuli,clamp_mode))

        return sweeps.sweep_number.values[-1]

    def get_sweep_record(self, sweep_number):
        """
        Parameters
        ----------
        sweep_number: int sweep number
        Returns
        -------
        sweep_record: dict of sweep properties
        """

        st = self.sweep_table

        if sweep_number is not None:
            mask = st[self.SWEEP_NUMBER] == sweep_number
            st = st[mask]

        return st.to_dict(orient='records')[0]

    def sweep(self, sweep_number):

        """
        Create an instance of the Sweep class with the data loaded from the from a file
        Parameters
        ----------
        sweep_number: int
        Returns
        -------
        sweep: Sweep object
        """

        sweep_data = self.get_sweep_data(sweep_number)
        sweep_record = self.get_sweep_record(sweep_number)
        sampling_rate = sweep_data['sampling_rate']
        dt = 1. / sampling_rate
        t = np.arange(0, len(sweep_data['stimulus'])) * dt

        epochs = sweep_data.get('epochs')
        clamp_mode = sweep_record['clamp_mode']

        if clamp_mode == "VoltageClamp":
            v = sweep_data['stimulus']
            i = sweep_data['response']
        elif clamp_mode == "CurrentClamp":
            v = sweep_data['response']
            i = sweep_data['stimulus']
        else:
            raise Exception("Unable to determine clamp mode for sweep " + sweep_number)

        v *= 1.0e3    # convert units V->mV
        i *= 1.0e12   # convert units A->pA

        if len(sweep_data['stimulus']) != len(sweep_data['response']):
            warnings.warn("Stimulus duration {} is not equal reponse duration {}".
                          format(len(sweep_data['stimulus']),len(sweep_data['response'])))

        try:
            sweep = Sweep(t=t,
                          v=v,
                          i=i,
                          sampling_rate=sampling_rate,
                          sweep_number=sweep_number,
                          clamp_mode=clamp_mode,
                          epochs=epochs,
                          )

        except Exception:
            logging.warning("Error reading sweep %d" % sweep_number)
            raise

        return sweep

    def sweep_set(self, sweep_numbers):
        try:
            return SweepSet([self.sweep(sn) for sn in sweep_numbers])
        except TypeError:  # not iterable
            return SweepSet([self.sweep(sweep_numbers)])

    def aligned_sweeps(self, sweep_numbers, stim_onset_delta):
        raise NotImplementedError


    def get_sweep_data(self, sweep_number):
        """
        Read sweep data from the nwb file
        Substitute trailing zeros in the response for np.nan
        because trailing zeros occur after the end of recording
        Parameters
        ----------
        sweep_number
        Returns
        -------
        dict in the format:
        {
            'stimulus': np.ndarray,
            'response': np.ndarray,
            'stimulus_unit': string,
            'sampling_rate': float
        }
        """
        self.abf_data.setSweep(sweep_number)
        sweep_data = {'stimulus': self.abf_data.sweepC, 'response': self.abf_data.sweepY, 'stimulus_unit': self.abf_data.sweepUnitsC, 'sampling_rate': self.abf_data.dataRate}

        response = sweep_data['response']

        if len(np.flatnonzero(response)) == 0:
            recording_end_idx = 0
            sweep_end_idx = 0
        else:
            recording_end_idx = np.flatnonzero(response)[-1]
            sweep_end_idx = len(response)-1

        if recording_end_idx < sweep_end_idx:
            response[recording_end_idx+1:] = np.nan

        return sweep_data

