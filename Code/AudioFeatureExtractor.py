import librosa
import numpy as np

class AudioFeatureExtractor():
    STATS = ['mean', 'std']
    TEMPO = 'tempo'
    INIT_FEATURE_MATRIX = ['amplitude_envelope', 'root_mean_square_energy', 'energy_entropy', 
                           'zero_crossing_rate', 'band_energy_ratio', 'spectral_centroid', 
                           'spectral_bandwidth', 'spectral_rolloff', 'spectral_flatness', 'spectral_contrast']
    MFCC = 'mfcc'
    N_MFCC = 13
    CHROMA = 'chroma'
    N_CHROMA = 12
    S_FLUX = 'spectral_flux'
    GENRE = 'genre'

    def __init__(self, frame_size=1024, hop_length=512):
        self.frame_size=frame_size
        self.hop_length=hop_length

### Generating the feature list 

    '''
    Generate the feature list (73 input features, 1 output feature)
    Order: 
        1. tempo
        2. Means: (ae, rms, ee, zcr, ber, s_centr, sb, s_roll, s_flat, s_contr, mfcc, chroma, s_flux)
        3. Standard Deviations: "
        4. genre
    '''
    def generate_feature_list(self):
        feature_list = [AudioFeatureExtractor.TEMPO]
        audio_features = self._generate_audio_feature_list()

        for stat in AudioFeatureExtractor.STATS:
            for feature in audio_features:
                feature_list.append((feature + '_' + stat))

        feature_list.append(AudioFeatureExtractor.GENRE)
        return np.array(feature_list)

    def _generate_audio_feature_list(self):
        audio_features = AudioFeatureExtractor.INIT_FEATURE_MATRIX.copy()

        for i in range(1, AudioFeatureExtractor.N_MFCC + 1):
            audio_features.append((AudioFeatureExtractor.MFCC + str(i)))

        for i in range(1, AudioFeatureExtractor.N_CHROMA + 1):
            audio_features.append((AudioFeatureExtractor.CHROMA + str(i)))

        audio_features.append(AudioFeatureExtractor.S_FLUX)
        return audio_features

### Feature Extraction

    def extract(self, filename, genre):
        signal, sr = librosa.load(filename)

        tempo = self._tempo(signal, sr)
        ae, rms, ee, zcr = self._extract_time_features(signal)
        ber, s_centr, sb, s_roll, s_flat, s_contr, s_flux = self._extract_frequency_features(signal, sr)
        mfcc = self._mfcc(signal, sr)
        chroma = self._chroma(signal, sr)

        feature_matrix = np.vstack([ae, rms, ee, zcr, ber, s_centr, sb, s_roll, s_flat, s_contr, mfcc, chroma])

        return self._compile_data_vector(tempo, feature_matrix, s_flux, genre)

    def _extract_time_features(self, signal):
        ae = self._amplitude_envelope(signal)
        rms = self._root_mean_square_energy(signal)
        ee = self._energy_entropy(signal)
        zcr = self._zero_crossing_rate(signal)
        return ae, rms, ee, zcr

    def _extract_frequency_features(self, signal, sr):
        spectrogram = self._extract_spectrogram(signal)

        ber = self._band_energy_ratio(spectrogram, sr)
        s_centr = self._spectral_centroid(signal, sr)
        sb = self._spectral_bandwidth(signal, sr)
        s_roll = self._spectral_rolloff(signal, sr)
        s_flat = self._spectral_flatness(signal)
        s_contr = self._spectral_contrast(signal, sr)
        s_flux = self._spectral_flux(spectrogram)

        return ber, s_centr, sb, s_roll, s_flat, s_contr, s_flux

    def _compile_data_vector(self, tempo, feature_matrix, s_flux, genre):
        means = np.mean(feature_matrix, axis=1)
        std_devs = np.std(feature_matrix, axis=1)
        return np.hstack([tempo, means, np.mean(s_flux), std_devs, np.std(s_flux), genre])

### Time-Domain Features

    # Tempo (beats per minute)
    def _tempo(self, signal, sr):
        return librosa.beat.tempo(y=signal, sr=sr, hop_length=self.hop_length)

    # Amplitude Envelope
    def _amplitude_envelope(self, signal):
        return np.array([max(signal[i:(i + self.frame_size)]) for i in range(0, len(signal), self.hop_length)])

    # Root-Mean Square Energy
    def _root_mean_square_energy(self, signal):
        return librosa.feature.rms(signal, frame_length=self.frame_size, hop_length=self.hop_length)[0]
    
    # Energy Entropy
    def _energy_entropy(self, signal):
        return np.array([self._compute_frame_energy_entropy(signal[i:(i + self.frame_size)]) for i in range(0, len(signal), self.hop_length)])

    def _compute_frame_energy_entropy(self, frame, num_subframes=10):
        subframe_size = int(np.floor(len(frame) / num_subframes))
        subframes = [frame[i:(i + subframe_size)] for i in range(0, len(frame), subframe_size)]
        
        energy = np.array([self._compute_energy(subframe) for subframe in subframes])
        energy = energy / np.sum(energy)
        
        return -np.sum(energy * np.log2(energy))

    def _compute_energy(self, frame):
        return np.sum(np.abs(frame)**2) / len(frame)

    # Zero Crossing Rate
    def _zero_crossing_rate(self, signal):
        return librosa.feature.zero_crossing_rate(signal, frame_length=self.frame_size, hop_length=self.hop_length)[0]

    
### Frequency-Domain Features

    def _extract_spectrogram(self, signal):
        return librosa.stft(signal, n_fft=self.frame_size, hop_length=self.hop_length)

    # Band Energy Ratio
    def _band_energy_ratio(self, spectrogram, sr, split_frequency=2000):
        split_frequency_bin = self._calculate_split_frequency_bin(spectrogram, split_frequency, sr)
        power_spec = self._to_power_spectrogram(spectrogram).T
        return np.array([self._calculate_ber_for_frame(freqs_in_frame, split_frequency_bin) for freqs_in_frame in power_spec])

    def _calculate_split_frequency_bin(self, spectrogram, split_frequency, sr):
        frequency_range = sr / 2
        frequency_delta_per_bin = frequency_range / spectrogram.shape[0]
        split_frequency_bin = np.floor(split_frequency / frequency_delta_per_bin)
        return int(split_frequency_bin)

    def _to_power_spectrogram(self, spectrogram):
        return (np.abs(spectrogram) ** 2)

    def _calculate_ber_for_frame(self, frequencies_in_frame, split_frequency_bin):
        sum_power_low_freq = np.sum(frequencies_in_frame[:split_frequency_bin])
        sum_power_high_freq = np.sum(frequencies_in_frame[split_frequency_bin:])
        return (sum_power_low_freq / sum_power_high_freq)

    # Spectral Centroid
    def _spectral_centroid(self, signal, sr):
        return librosa.feature.spectral_centroid(signal, sr=sr, n_fft=self.frame_size, hop_length=self.hop_length)[0]

    # Spectral Bandwidth
    def _spectral_bandwidth(self, signal, sr):
        return librosa.feature.spectral_bandwidth(signal, sr=sr, n_fft=self.frame_size, hop_length=self.hop_length)[0]

    # Spectral Rolloff
    def _spectral_rolloff(self, signal, sr):
        return librosa.feature.spectral_rolloff(signal, sr=sr, n_fft=self.frame_size, hop_length=self.hop_length)[0]

    # Spectral Flatness
    def _spectral_flatness(self, signal):
        return librosa.feature.spectral_flatness(signal, n_fft=self.frame_size, hop_length=self.hop_length)[0]

    # Spectral Contrast
    def _spectral_contrast(self, signal, sr):
        return librosa.feature.spectral_contrast(signal, sr=sr, n_fft=self.frame_size, hop_length=self.hop_length)[0]

    # Spectral Flux
    def _spectral_flux(self, spectrogram):
        num_frames = spectrogram.shape[1]
        curr_spectra = spectrogram[:, 0]
        curr_spectra_nmlzd = self._normalize(curr_spectra)

        sf = []
        for i in range(1, num_frames):
            frame_sf, curr_spectra_nmlzd = self._compute_spect_flux_for_frame(curr_spectra_nmlzd, spectrogram[:, i])
            sf.append(frame_sf)
            
        return np.array(sf)

    def _normalize(self, spectra):
        return spectra / np.sum(spectra)

    def _compute_spect_flux_for_frame(self, prev_spectra_nmlzd, curr_spectra):
        curr_spectra_nmlzd = self._normalize(curr_spectra)
        frame_sf = np.linalg.norm(curr_spectra_nmlzd - prev_spectra_nmlzd)
        return frame_sf, curr_spectra_nmlzd

### Mel-Frequency Cepstral Coefficients
    def _mfcc(self, signal, sr):
        return librosa.feature.mfcc(signal, n_mfcc=AudioFeatureExtractor.N_MFCC, sr=sr)

### Chroma Vector
    def _chroma(self, signal, sr):
        return librosa.feature.chroma_stft(signal, n_chroma=AudioFeatureExtractor.N_CHROMA, sr=sr, n_fft=self.frame_size, hop_length=self.hop_length)