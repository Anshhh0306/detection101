"""
Audio Feature Extraction Module for AI Voice Detection - VERSION 3
==================================================================
ADVANCED FEATURES including:
1. Phase coherence analysis (WaveNet/Tacotron artifact detection)
2. Formant transitions smoothness
3. Enhanced jitter/shimmer (RAP, PPQ, APQ)
4. Pause pattern analysis (breathing detection)
5. Filler word detection patterns (uh, umm energy signatures)
6. Spectrogram artifact detection
7. Prosody naturalness metrics

Feature count: 200 features for robust AI detection
"""

import librosa
import numpy as np
from typing import Optional, Tuple, List
from scipy import signal
from scipy.stats import kurtosis, skew, entropy
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert, find_peaks, medfilt
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Configuration
SAMPLE_RATE = 22050
MAX_DURATION = 120  # Allow up to 2 minutes for multi-pass analysis
CHUNK_DURATION = 20  # Each chunk is 20 seconds
N_FEATURES = 225  # V4: Added 25 background/emotion features


def extract_features(audio_path: str, duration: Optional[float] = MAX_DURATION) -> np.ndarray:
    """
    Extract 200 advanced features for AI voice detection.
    
    Features target specific artifacts of:
    - Neural TTS (WaveNet, Tacotron, FastSpeech)
    - Concatenative TTS
    - Voice cloning systems
    """
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=duration, mono=True)
        
        # Ensure minimum length
        if len(y) < sr * 0.5:
            y = np.pad(y, (0, int(sr * 0.5) - len(y)), mode='constant')
        
        # Normalize audio
        y = y / (np.max(np.abs(y)) + 1e-8)
        
        features = []
        
        # ========== PART 1: MFCC & SPECTRAL BASICS (40 features) ==========
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        features.extend(np.mean(mfccs, axis=1))      # 13
        features.extend(np.std(mfccs, axis=1))       # 13
        
        delta_mfccs = librosa.feature.delta(mfccs)
        features.extend(np.mean(delta_mfccs, axis=1))  # 13
        
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.append(np.mean(spec_centroid))  # 1
        # Total: 40
        
        # ========== PART 2: ENHANCED PITCH ANALYSIS (30 features) ==========
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=50, fmax=500, sr=sr, fill_na=0.0
        )
        
        f0_voiced = f0[f0 > 0]
        
        if len(f0_voiced) > 10:
            # Basic pitch stats
            features.append(np.mean(f0_voiced))
            features.append(np.std(f0_voiced))
            features.append(np.max(f0_voiced) - np.min(f0_voiced))
            features.append(np.median(f0_voiced))
            
            # === JITTER ANALYSIS (Multiple measures) ===
            pitch_diff = np.abs(np.diff(f0_voiced))
            
            # Local Jitter (Jitter local)
            jitter_local = 100 * np.mean(pitch_diff) / np.mean(f0_voiced)
            features.append(jitter_local)
            
            # RAP - Relative Average Perturbation (3-point average)
            rap_values = []
            for i in range(1, len(f0_voiced) - 1):
                avg = (f0_voiced[i-1] + f0_voiced[i] + f0_voiced[i+1]) / 3
                rap_values.append(abs(f0_voiced[i] - avg))
            rap = 100 * np.mean(rap_values) / np.mean(f0_voiced) if rap_values else 0
            features.append(rap)
            
            # PPQ5 - Period Perturbation Quotient (5-point average)
            ppq_values = []
            for i in range(2, len(f0_voiced) - 2):
                avg = np.mean(f0_voiced[i-2:i+3])
                ppq_values.append(abs(f0_voiced[i] - avg))
            ppq5 = 100 * np.mean(ppq_values) / np.mean(f0_voiced) if ppq_values else 0
            features.append(ppq5)
            
            # DDP - Difference of Differences of Periods
            if len(pitch_diff) > 1:
                ddp = np.mean(np.abs(np.diff(pitch_diff)))
                ddp_rel = 100 * ddp / np.mean(f0_voiced)
                features.append(ddp_rel)
            else:
                features.append(0.0)
            
            # Jitter std
            features.append(np.std(pitch_diff))
            
            # === INTONATION CONTOUR ANALYSIS ===
            # Smoothed pitch contour
            smoothed_f0 = uniform_filter1d(f0_voiced, size=5)
            
            # Pitch contour roughness (AI is TOO smooth)
            roughness = np.mean(np.abs(f0_voiced - smoothed_f0))
            features.append(roughness)
            
            # Pitch slope changes (inflection points) - humans have more varied intonation
            pitch_slopes = np.diff(smoothed_f0)
            sign_changes = np.sum(np.abs(np.diff(np.sign(pitch_slopes)))) / 2
            features.append(sign_changes / len(pitch_slopes) if len(pitch_slopes) > 0 else 0)
            
            # Pitch range usage
            pitch_range = np.max(f0_voiced) - np.min(f0_voiced)
            features.append(pitch_range / np.mean(f0_voiced))  # Normalized range
            
            # Pitch percentiles
            features.append(np.percentile(f0_voiced, 25))
            features.append(np.percentile(f0_voiced, 75))
            features.append(np.percentile(f0_voiced, 90) - np.percentile(f0_voiced, 10))
            
            # Higher order moments
            features.append(skew(f0_voiced))
            features.append(kurtosis(f0_voiced))
            
            # Pitch entropy
            hist, _ = np.histogram(f0_voiced, bins=20, density=True)
            features.append(entropy(hist + 1e-10))
            
            # Local pitch variation
            local_vars = [np.std(f0_voiced[i:i+10]) for i in range(0, len(f0_voiced)-10, 5)]
            features.append(np.mean(local_vars) if local_vars else 0)
            features.append(np.std(local_vars) if len(local_vars) > 1 else 0)
            
            # Pitch acceleration
            if len(pitch_diff) > 1:
                pitch_accel = np.diff(pitch_diff)
                features.append(np.mean(np.abs(pitch_accel)))
                features.append(np.std(pitch_accel))
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0] * 22)
        
        # Voiced ratio
        voiced_ratio = np.sum(f0 > 0) / (len(f0) + 1e-6)
        features.append(voiced_ratio)
        
        # Voiced segment analysis
        voiced_segments = []
        seg_len = 0
        for v in (f0 > 0):
            if v:
                seg_len += 1
            elif seg_len > 0:
                voiced_segments.append(seg_len)
                seg_len = 0
        if seg_len > 0:
            voiced_segments.append(seg_len)
        
        if len(voiced_segments) > 2:
            features.append(np.mean(voiced_segments))
            features.append(np.std(voiced_segments))
            features.append(len(voiced_segments))  # Number of voiced segments
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Vibrato detection
        if len(f0_voiced) > 30:
            f0_detrend = f0_voiced - uniform_filter1d(f0_voiced, size=15)
            autocorr = np.correlate(f0_detrend, f0_detrend, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            if autocorr[0] > 0:
                autocorr = autocorr / autocorr[0]
                features.append(np.max(autocorr[5:30]) if len(autocorr) > 30 else 0)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Pad to 30
        while len(features) < 70:
            features.append(0.0)
        # Total Part 2: 30 (cumulative: 70)
        
        # ========== PART 3: PAUSE & BREATHING ANALYSIS (25 features) ==========
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        
        features.append(np.mean(rms))
        features.append(np.std(rms))
        features.append(np.max(rms) - np.min(rms))
        
        # Silence detection
        silence_thresh = np.mean(rms) * 0.15
        is_silence = rms < silence_thresh
        
        # Pause analysis (KEY: humans pause for breath)
        pauses = []
        pause_len = 0
        pause_positions = []  # Track where pauses occur
        for i, s in enumerate(is_silence):
            if s:
                pause_len += 1
            elif pause_len > 0:
                pauses.append(pause_len)
                pause_positions.append(i - pause_len)
                pause_len = 0
        if pause_len > 0:
            pauses.append(pause_len)
            pause_positions.append(len(is_silence) - pause_len)
        
        if len(pauses) > 0:
            features.append(len(pauses))
            features.append(np.mean(pauses))
            features.append(np.std(pauses) if len(pauses) > 1 else 0)
            features.append(np.max(pauses))
            features.append(np.min(pauses))
            features.append(np.median(pauses))
            
            # Pause distribution entropy (humans have more varied pauses)
            if len(pauses) > 3:
                hist, _ = np.histogram(pauses, bins=min(10, len(pauses)), density=True)
                features.append(entropy(hist + 1e-10))
            else:
                features.append(0.0)
            
            # Inter-pause intervals (rhythm of pauses)
            if len(pause_positions) > 1:
                inter_pause = np.diff(pause_positions)
                features.append(np.mean(inter_pause))
                features.append(np.std(inter_pause))
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0] * 9)
        
        # Silence ratio
        features.append(np.sum(is_silence) / (len(is_silence) + 1e-6))
        
        # === BREATHING PATTERN DETECTION ===
        # Breathing often appears as low energy with specific spectral signature
        # Look for short (~0.3-0.8s) low-energy segments before speech
        breath_candidates = 0
        for i, pl in enumerate(pauses):
            # Check if pause is in breathing range (3-8 frames = ~0.15-0.4s)
            if 3 <= pl <= 8:
                breath_candidates += 1
        features.append(breath_candidates)
        
        # Energy contour smoothness
        energy_diff = np.abs(np.diff(rms))
        features.append(np.mean(energy_diff))
        features.append(np.std(energy_diff))
        
        # Energy acceleration
        if len(energy_diff) > 1:
            energy_accel = np.abs(np.diff(energy_diff))
            features.append(np.mean(energy_accel))
        else:
            features.append(0.0)
        
        # Attack characteristics
        attacks = np.diff(rms)
        positive_attacks = attacks[attacks > 0]
        if len(positive_attacks) > 0:
            features.append(np.mean(positive_attacks))
            features.append(np.max(positive_attacks))
        else:
            features.extend([0.0, 0.0])
        
        # Speech rate
        transitions = np.sum(np.abs(np.diff(is_silence.astype(int))))
        features.append(transitions / (len(is_silence) + 1e-6))
        
        # Energy entropy
        rms_norm = rms / (np.sum(rms) + 1e-10)
        features.append(entropy(rms_norm + 1e-10))
        
        # Pad to 25
        while len(features) < 95:
            features.append(0.0)
        # Total Part 3: 25 (cumulative: 95)
        
        # ========== PART 4: SHIMMER & AMPLITUDE PERTURBATION (20 features) ==========
        peaks, peak_props = find_peaks(np.abs(y), distance=int(sr/400), height=0.01)
        
        if len(peaks) > 20:
            peak_amps = np.abs(y[peaks])
            
            # Shimmer local
            amp_diff = np.abs(np.diff(peak_amps))
            shimmer_local = 100 * np.mean(amp_diff) / np.mean(peak_amps)
            features.append(shimmer_local)
            
            # Shimmer dB
            shimmer_db = np.mean(np.abs(20 * np.log10(peak_amps[1:] / (peak_amps[:-1] + 1e-10) + 1e-10)))
            features.append(shimmer_db)
            
            # APQ3 - Amplitude Perturbation Quotient (3-point)
            apq3_vals = []
            for i in range(1, len(peak_amps) - 1):
                avg = np.mean(peak_amps[i-1:i+2])
                apq3_vals.append(abs(peak_amps[i] - avg))
            apq3 = 100 * np.mean(apq3_vals) / np.mean(peak_amps) if apq3_vals else 0
            features.append(apq3)
            
            # APQ5 - Amplitude Perturbation Quotient (5-point)
            apq5_vals = []
            for i in range(2, len(peak_amps) - 2):
                avg = np.mean(peak_amps[i-2:i+3])
                apq5_vals.append(abs(peak_amps[i] - avg))
            apq5 = 100 * np.mean(apq5_vals) / np.mean(peak_amps) if apq5_vals else 0
            features.append(apq5)
            
            # APQ11 - Amplitude Perturbation Quotient (11-point)
            apq11_vals = []
            for i in range(5, len(peak_amps) - 5):
                avg = np.mean(peak_amps[i-5:i+6])
                apq11_vals.append(abs(peak_amps[i] - avg))
            apq11 = 100 * np.mean(apq11_vals) / np.mean(peak_amps) if apq11_vals else 0
            features.append(apq11)
            
            # DDA - Difference of Differences of Amplitudes
            if len(amp_diff) > 1:
                dda = np.mean(np.abs(np.diff(amp_diff)))
                dda_rel = 100 * dda / np.mean(peak_amps)
                features.append(dda_rel)
            else:
                features.append(0.0)
            
            # Amplitude std and variation
            features.append(np.std(peak_amps))
            features.append(np.std(amp_diff))
        else:
            features.extend([0.0] * 8)
        
        # Zero-crossing based analysis
        zc = np.where(np.diff(np.signbit(y)))[0]
        if len(zc) > 20:
            zc_intervals = np.diff(zc)
            features.append(np.mean(zc_intervals))
            features.append(np.std(zc_intervals))
            features.append(np.std(zc_intervals) / (np.mean(zc_intervals) + 1e-6))  # Coefficient of variation
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Harmonic analysis
        harmonic = librosa.effects.harmonic(y)
        percussive = librosa.effects.percussive(y)
        noise = y - harmonic
        
        # HNR - Harmonics to Noise Ratio
        hnr = 10 * np.log10((np.sum(harmonic**2) + 1e-10) / (np.sum(noise**2) + 1e-10))
        features.append(hnr)
        
        # NHR - Noise to Harmonics Ratio
        nhr = np.sum(noise**2) / (np.sum(harmonic**2) + 1e-10)
        features.append(nhr)
        
        # Harmonic-percussive ratio
        hp_ratio = np.sum(harmonic**2) / (np.sum(percussive**2) + 1e-10)
        features.append(np.log10(hp_ratio + 1e-10))
        
        # Noise characteristics
        features.append(np.mean(np.abs(noise)))
        features.append(np.std(noise))
        
        # Pad to 20
        while len(features) < 115:
            features.append(0.0)
        # Total Part 4: 20 (cumulative: 115)
        
        # ========== PART 5: PHASE COHERENCE & SYNTHESIS ARTIFACTS (25 features) ==========
        # This section detects artifacts from neural TTS systems
        
        # Compute STFT with phase
        D = librosa.stft(y, n_fft=2048, hop_length=512)
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # Phase coherence analysis
        # AI systems often have unnaturally smooth phase transitions
        phase_diff = np.diff(phase, axis=1)
        
        # Unwrap phase differences
        phase_diff_unwrapped = np.unwrap(phase_diff, axis=1)
        
        # Phase smoothness (AI = too smooth)
        phase_smoothness = np.mean(np.std(phase_diff_unwrapped, axis=1))
        features.append(phase_smoothness)
        
        # Phase discontinuities - look for sudden jumps
        phase_jumps = np.abs(phase_diff_unwrapped) > np.pi/2
        phase_discontinuity_rate = np.mean(phase_jumps)
        features.append(phase_discontinuity_rate)
        
        # Group delay analysis
        # Group delay is derivative of phase - AI often has unnatural patterns
        if phase_diff_unwrapped.shape[1] > 0:
            group_delay = np.mean(phase_diff_unwrapped, axis=0)
            features.append(np.mean(group_delay))
            features.append(np.std(group_delay))
            features.append(skew(group_delay) if len(group_delay) > 2 else 0)
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Instantaneous frequency analysis
        analytic_signal = hilbert(np.asarray(y))
        instantaneous_phase = np.unwrap(np.angle(np.asarray(analytic_signal)))
        instantaneous_freq = np.diff(instantaneous_phase) / (2.0 * np.pi) * sr
        
        # IF stability (AI tends to have more stable IF)
        if len(instantaneous_freq) > 0:
            features.append(np.std(instantaneous_freq))
            features.append(np.mean(np.abs(np.diff(instantaneous_freq))))
        else:
            features.extend([0.0, 0.0])
        
        # Spectral flux with phase
        spec_flux = np.sqrt(np.sum(np.diff(magnitude, axis=1)**2, axis=0))
        features.append(np.mean(spec_flux))
        features.append(np.std(spec_flux))
        features.append(skew(spec_flux) if len(spec_flux) > 2 else 0)
        
        # Spectral flatness variation
        spec_flat = librosa.feature.spectral_flatness(y=y)[0]
        features.append(np.mean(spec_flat))
        features.append(np.std(spec_flat))
        
        # === FORMANT TRANSITION ANALYSIS ===
        # AI often has unnaturally smooth formant transitions
        # Use LPC for formant estimation
        try:
            from scipy.signal import lfilter
            frame_length = int(0.025 * sr)  # 25ms frames
            hop = int(0.010 * sr)  # 10ms hop
            
            formant_tracks = []
            for i in range(0, len(y) - frame_length, hop):
                frame = y[i:i + frame_length]
                frame = frame * np.hamming(len(frame))
                
                # LPC analysis
                lpc_order = 2 + sr // 1000
                lpc_order = min(lpc_order, len(frame) - 1)
                
                autocorr = np.correlate(frame, frame, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                if len(autocorr) > lpc_order:
                    R = autocorr[:lpc_order + 1]
                    # Solve Yule-Walker equations
                    try:
                        from scipy.linalg import solve_toeplitz
                        a = solve_toeplitz((R[:-1], R[:-1]), -R[1:])
                        
                        # Find formants from LPC roots
                        roots = np.roots(np.append(1, a))
                        roots = roots[np.imag(roots) > 0]
                        
                        if len(roots) > 0:
                            angles = np.angle(roots)
                            freqs = angles * (sr / (2 * np.pi))
                            freqs = sorted(freqs[(freqs > 50) & (freqs < 5000)])
                            if len(freqs) >= 2:
                                formant_tracks.append([freqs[0], freqs[1]])
                    except:
                        pass
            
            if len(formant_tracks) > 10:
                formant_tracks = np.array(formant_tracks)
                # F1 and F2 dynamics
                f1_track = formant_tracks[:, 0]
                f2_track = formant_tracks[:, 1]
                
                # Formant transition smoothness (AI = too smooth)
                f1_diff = np.diff(f1_track)
                f2_diff = np.diff(f2_track)
                
                features.append(np.std(f1_diff))  # F1 transition variability
                features.append(np.std(f2_diff))  # F2 transition variability
                
                # Formant movement range
                features.append(np.max(f1_track) - np.min(f1_track))
                features.append(np.max(f2_track) - np.min(f2_track))
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        except:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Spectral bandwidth dynamics
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features.append(np.mean(spec_bw))
        features.append(np.std(spec_bw))
        
        # Spectral rolloff
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features.append(np.mean(spec_roll))
        features.append(np.std(spec_roll))
        
        # Pad to 25
        while len(features) < 140:
            features.append(0.0)
        # Total Part 5: 25 (cumulative: 140)
        
        # ========== PART 6: TEMPORAL DYNAMICS & PROSODY (25 features) ==========
        
        # MFCC dynamics
        mfcc_diff = np.diff(mfccs, axis=1)
        features.append(np.mean(np.abs(mfcc_diff)))
        features.append(np.std(np.abs(mfcc_diff)))
        
        # Delta-delta MFCCs
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        features.append(np.mean(np.abs(delta2_mfccs)))
        features.append(np.std(np.abs(delta2_mfccs)))
        
        # Temporal flatness
        frame_energy = np.sum(magnitude**2, axis=0) + 1e-10
        geometric_mean = np.exp(np.mean(np.log(frame_energy)))
        arithmetic_mean = np.mean(frame_energy)
        features.append(geometric_mean / arithmetic_mean)
        
        # Modulation spectrum - captures rhythm patterns
        if len(rms) > 50:
            mod_spec = np.abs(np.fft.fft(rms - np.mean(rms)))[:len(rms)//2]
            features.append(np.argmax(mod_spec[1:20]) + 1)  # Dominant modulation frequency
            features.append(np.max(mod_spec[1:20]))
            features.append(np.sum(mod_spec[1:10]) / (np.sum(mod_spec[10:20]) + 1e-10))  # Low/high mod ratio
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Spectral contrast (peak vs valley)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
        features.append(np.mean(spec_contrast))
        features.append(np.std(spec_contrast))
        
        # Long-term dynamics - look at 1-second windows
        window_size = int(sr)
        hop_size = int(sr // 2)
        long_term_energy = []
        for i in range(0, len(y) - window_size, hop_size):
            long_term_energy.append(np.sum(y[i:i+window_size]**2))
        
        if len(long_term_energy) > 2:
            features.append(np.std(long_term_energy))
            features.append(np.max(long_term_energy) / (np.min(long_term_energy) + 1e-10))
        else:
            features.extend([0.0, 0.0])
        
        # Rhythm regularity
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # Handle tempo - may be array in newer librosa versions
        tempo_val = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        features.append(tempo_val)
        
        if len(beats) > 2:
            beat_intervals = np.diff(beats)
            features.append(np.std(beat_intervals))  # Rhythm regularity (AI = more regular)
        else:
            features.append(0.0)
        
        # Onset detection
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        features.append(len(onsets))  # Number of onsets
        
        if len(onsets) > 2:
            onset_intervals = np.diff(onsets)
            features.append(np.std(onset_intervals))
        else:
            features.append(0.0)
        
        # Spectral entropy over time
        spec_norm = magnitude / (np.sum(magnitude, axis=0, keepdims=True) + 1e-10)
        spectral_entropy = -np.sum(spec_norm * np.log2(spec_norm + 1e-10), axis=0)
        features.append(np.mean(spectral_entropy))
        features.append(np.std(spectral_entropy))
        
        # Pad to 25
        while len(features) < 165:
            features.append(0.0)
        # Total Part 6: 25 (cumulative: 165)
        
        # ========== PART 7: FILLER WORD PATTERNS & NATURAL SPEECH (20 features) ==========
        # Detect patterns typical of filler words (uh, um, hmm)
        # These have specific energy/spectral signatures
        
        # Energy-based segment analysis
        frames = librosa.util.frame(y, frame_length=2048, hop_length=512)
        frame_energies = np.sum(frames**2, axis=0)
        
        # Look for low-intensity sustained segments (potential fillers)
        low_energy_thresh = np.percentile(frame_energies, 30)
        med_energy_thresh = np.percentile(frame_energies, 60)
        
        potential_fillers = []
        filler_len = 0
        for e in frame_energies:
            if low_energy_thresh < e < med_energy_thresh:
                filler_len += 1
            elif filler_len > 3:  # At least ~0.15s
                potential_fillers.append(filler_len)
                filler_len = 0
            else:
                filler_len = 0
        
        features.append(len(potential_fillers))
        features.append(np.mean(potential_fillers) if potential_fillers else 0)
        
        # Spectral characteristics of sustained segments
        spec_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Low centroid sustained segments (often fillers)
        low_centroid_thresh = np.percentile(spec_centroids, 25)
        low_centroid_sustained = 0
        sustained_len = 0
        for c in spec_centroids:
            if c < low_centroid_thresh:
                sustained_len += 1
            elif sustained_len > 5:
                low_centroid_sustained += 1
                sustained_len = 0
            else:
                sustained_len = 0
        features.append(low_centroid_sustained)
        
        # Speech rate variation (humans vary more)
        if len(rms) > 20:
            # Local speech rate using energy transitions
            local_rates = []
            for i in range(0, len(rms) - 20, 10):
                local_transitions = np.sum(np.abs(np.diff((rms[i:i+20] > silence_thresh).astype(int))))
                local_rates.append(local_transitions)
            features.append(np.std(local_rates) if local_rates else 0)
        else:
            features.append(0.0)
        
        # Hesitation patterns
        # Look for quick energy drops followed by recovery
        hesitations = 0
        for i in range(2, len(rms) - 2):
            if rms[i] < rms[i-1] * 0.5 and rms[i] < rms[i+1] * 0.5:
                hesitations += 1
        features.append(hesitations)
        
        # Spectral stability during low-energy segments
        if len(potential_fillers) > 0 and len(spec_centroids) > 10:
            # Find indices of potential fillers and check spectral stability
            filler_stability = np.std(spec_centroids[spec_centroids < np.median(spec_centroids)])
            features.append(filler_stability)
        else:
            features.append(0.0)
        
        # Micro-pause patterns (< 150ms)
        micro_pauses = [p for p in pauses if 1 <= p <= 3]
        features.append(len(micro_pauses))
        features.append(len(micro_pauses) / (len(pauses) + 1e-6) if pauses else 0)
        
        # Breath-like patterns before speech
        breath_patterns = 0
        for i in range(1, len(pauses)):
            if pauses[i] <= 4 and pauses[i-1] >= 5:  # Short pause after long one
                breath_patterns += 1
        features.append(breath_patterns)
        
        # Energy variation in voiced segments
        voiced_energies = rms[~is_silence] if np.sum(~is_silence) > 0 else rms
        features.append(np.std(voiced_energies))
        features.append(skew(voiced_energies) if len(voiced_energies) > 2 else 0)
        
        # Speaking rate irregularity
        if len(onsets) > 3:
            onset_intervals = np.diff(onsets)
            features.append(np.std(onset_intervals) / (np.mean(onset_intervals) + 1e-6))  # CV
        else:
            features.append(0.0)
        
        # Chroma features for tonal content
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.append(np.std(np.mean(chroma, axis=1)))
        features.append(entropy(np.mean(chroma, axis=1) + 1e-10))
        
        # Pad to 20
        while len(features) < 185:
            features.append(0.0)
        # Total Part 7: 20 (cumulative: 185)
        
        # ========== PART 8: ADDITIONAL ROBUSTNESS FEATURES (15 features) ==========
        
        # Log mel spectrogram statistics
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
        log_mel = librosa.power_to_db(mel_spec)
        
        features.append(np.mean(log_mel))
        features.append(np.std(log_mel))
        features.append(np.mean(np.diff(log_mel, axis=1)))
        
        # Temporal variation per mel band
        mel_temporal_var = np.std(log_mel, axis=1)
        features.append(np.mean(mel_temporal_var))
        features.append(np.std(mel_temporal_var))
        
        # Spectral decay characteristics
        if magnitude.shape[1] > 0:
            spectral_decay = np.mean(np.diff(np.mean(magnitude, axis=1)))
            features.append(spectral_decay)
        else:
            features.append(0.0)
        
        # Cepstral peak prominence (voice quality)
        if len(y) > 2048:
            cepstrum = np.fft.ifft(np.log(np.abs(np.fft.fft(y[:2048])) + 1e-10)).real
            cepstral_peak = np.max(cepstrum[int(sr/500):int(sr/50)])
            features.append(cepstral_peak)
        else:
            features.append(0.0)
        
        # Sub-band energy ratios
        low_band = np.sum(magnitude[:256, :]**2)
        mid_band = np.sum(magnitude[256:512, :]**2)
        high_band = np.sum(magnitude[512:, :]**2)
        total = low_band + mid_band + high_band + 1e-10
        
        features.append(low_band / total)
        features.append(mid_band / total)
        features.append(high_band / total)
        
        # Final statistics
        features.append(np.mean(np.abs(y)))
        features.append(np.std(y))
        features.append(kurtosis(y) if len(y) > 3 else 0)
        features.append(skew(y) if len(y) > 2 else 0)
        # Total Part 8: 15 (cumulative: 200)
        
        # ========== PART 9: V4 BACKGROUND, SMOOTHNESS & EMOTION (25 features) ==========
        
        # --- BACKGROUND ANALYSIS (8 features) ---
        # Find silent/quiet segments (AI has uniform noise floor, humans vary)
        try:
            silent_intervals = librosa.effects.split(y, top_db=30)
            if len(silent_intervals) > 1:
                noise_floors = []
                for start, end in silent_intervals:
                    if start > 0:
                        # Analyze the gap before this voiced segment
                        gap_start = silent_intervals[np.where(silent_intervals[:, 1] <= start)[0][-1]][1] if np.any(silent_intervals[:, 1] <= start) else 0
                        if start - gap_start > 100:
                            noise_floors.append(np.mean(np.abs(y[gap_start:start])))
                noise_floor_variance = np.std(noise_floors) if len(noise_floors) > 1 else 0.0
                noise_floor_mean = np.mean(noise_floors) if noise_floors else 0.0
            else:
                noise_floor_variance = 0.0
                noise_floor_mean = 0.0
        except:
            noise_floor_variance = 0.0
            noise_floor_mean = 0.0
        features.append(float(noise_floor_variance))  # AI = low variance
        features.append(float(noise_floor_mean))
        
        # SNR estimate (AI often has artificially high/consistent SNR)
        signal_power = np.mean(y**2)
        noise_estimate = np.percentile(np.abs(y), 10)**2 + 1e-10
        snr_estimate = 10 * np.log10(signal_power / noise_estimate + 1e-10)
        features.append(float(snr_estimate))
        
        # Background spectral consistency (AI backgrounds are suspiciously uniform)
        try:
            n_segments = min(5, magnitude.shape[1] // 10)
            if n_segments > 1:
                seg_len = magnitude.shape[1] // n_segments
                bg_spectra = []
                for i in range(n_segments):
                    seg_mag = magnitude[:, i*seg_len:(i+1)*seg_len]
                    bg_spectra.append(np.mean(seg_mag, axis=1))
                bg_consistency = np.mean([np.corrcoef(bg_spectra[i], bg_spectra[i+1])[0,1] 
                                          for i in range(len(bg_spectra)-1)])
                features.append(float(bg_consistency) if not np.isnan(bg_consistency) else 0.0)
            else:
                features.append(0.0)
        except:
            features.append(0.0)
        
        # Low frequency room tone (<100Hz) - real recordings have HVAC/room rumble
        low_freq_bins = int(100 * len(magnitude) / (sr/2))
        low_freq_energy = np.mean(magnitude[:low_freq_bins, :]**2) if low_freq_bins > 0 else 0.0
        features.append(float(low_freq_energy))
        
        # High frequency noise texture (>8kHz) - AI lacks natural hiss
        high_freq_start = int(8000 * len(magnitude) / (sr/2))
        high_freq_energy = np.mean(magnitude[high_freq_start:, :]**2) if high_freq_start < len(magnitude) else 0.0
        features.append(float(high_freq_energy))
        
        # Background transient count (real recordings have random clicks/bumps)
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            low_energy_mask = rms.flatten() < np.percentile(rms, 30)
            bg_onsets = onset_env[:len(low_energy_mask)][low_energy_mask[:len(onset_env)]]
            bg_transients = len(find_peaks(bg_onsets, height=np.mean(onset_env)*0.5)[0])
            features.append(float(bg_transients))
        except:
            features.append(0.0)
        
        # Background energy variance
        try:
            low_rms_frames = rms.flatten()[rms.flatten() < np.percentile(rms, 25)]
            bg_energy_var = np.std(low_rms_frames) if len(low_rms_frames) > 1 else 0.0
            features.append(float(bg_energy_var))
        except:
            features.append(0.0)
        
        # --- MULTI-SCALE SMOOTHNESS (8 features) ---
        # Multi-scale jitter (human jitter varies by scale, AI is constant)
        if f0 is not None and len(f0[f0 > 0]) > 50:
            f0_voiced = f0[f0 > 0]
            scales = [5, 10, 20, 50]
            scale_jitters = []
            for scale in scales:
                if len(f0_voiced) > scale:
                    smoothed = uniform_filter1d(f0_voiced, size=scale)
                    scale_jitter = np.std(f0_voiced - smoothed) / (np.mean(f0_voiced) + 1e-10)
                    scale_jitters.append(scale_jitter)
            jitter_scale_variance = np.std(scale_jitters) if len(scale_jitters) > 1 else 0.0
            jitter_scale_mean = np.mean(scale_jitters) if scale_jitters else 0.0
        else:
            jitter_scale_variance = 0.0
            jitter_scale_mean = 0.0
        features.append(float(jitter_scale_variance))  # AI = low
        features.append(float(jitter_scale_mean))
        
        # Shimmer temporal variance (AI shimmer is constant, human varies with emotion)
        try:
            n_windows = min(10, len(y) // (sr // 2))
            if n_windows > 2:
                window_len = len(y) // n_windows
                window_shimmers = []
                for i in range(n_windows):
                    win_y = y[i*window_len:(i+1)*window_len]
                    peaks, _ = find_peaks(np.abs(win_y), distance=int(sr/500))
                    if len(peaks) > 2:
                        amplitudes = np.abs(win_y[peaks])
                        win_shimmer = np.mean(np.abs(np.diff(amplitudes))) / (np.mean(amplitudes) + 1e-10)
                        window_shimmers.append(win_shimmer)
                shimmer_temporal_var = np.std(window_shimmers) if len(window_shimmers) > 1 else 0.0
            else:
                shimmer_temporal_var = 0.0
        except:
            shimmer_temporal_var = 0.0
        features.append(float(shimmer_temporal_var))
        
        # Amplitude micro-variation (tiny RMS fluctuations in ALL human speech)
        try:
            micro_rms = librosa.feature.rms(y=y, frame_length=256, hop_length=64)[0]
            micro_var = np.std(np.diff(micro_rms))
            features.append(float(micro_var))
        except:
            features.append(0.0)
        
        # ZCR consistency in voiced regions (AI has too-consistent ZCR)
        try:
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=256)[0]
            # Find voiced frames (higher energy)
            frame_energy = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
            voiced_mask = frame_energy > np.percentile(frame_energy, 40)
            voiced_zcr = zcr[:len(voiced_mask)][voiced_mask[:len(zcr)]]
            zcr_voiced_var = np.std(voiced_zcr) if len(voiced_zcr) > 1 else 0.0
            features.append(float(zcr_voiced_var))
        except:
            features.append(0.0)
        
        # Phoneme transition smoothness (frame-by-frame MFCC distance)
        try:
            mfcc_diff = np.diff(mfccs, axis=1)
            mfcc_distances = np.linalg.norm(mfcc_diff, axis=0)
            transition_smoothness = np.std(mfcc_distances)  # AI = low std, too smooth
            transition_peaks = len(find_peaks(mfcc_distances, height=np.mean(mfcc_distances)*1.5)[0])
            features.append(float(transition_smoothness))
            features.append(float(transition_peaks))
        except:
            features.append(0.0)
            features.append(0.0)
        
        # --- EMOTION & PROSODY (9 features) ---
        # Pitch-energy correlation (humans naturally couple pitch and loudness)
        if f0 is not None and len(f0[f0 > 0]) > 10:
            f0_voiced = f0[f0 > 0]
            # Interpolate to match RMS length
            rms_flat = rms.flatten()
            f0_interp = np.interp(np.linspace(0, 1, len(rms_flat)), 
                                  np.linspace(0, 1, len(f0_voiced)), f0_voiced)
            pitch_energy_corr = np.corrcoef(f0_interp, rms_flat)[0, 1]
            pitch_energy_corr = pitch_energy_corr if not np.isnan(pitch_energy_corr) else 0.0
        else:
            pitch_energy_corr = 0.0
        features.append(float(pitch_energy_corr))  # Humans = higher correlation
        
        # F0 declination (natural pitch drop across phrases)
        if f0 is not None and len(f0[f0 > 0]) > 20:
            f0_voiced = f0[f0 > 0]
            x = np.arange(len(f0_voiced))
            slope, _ = np.polyfit(x, f0_voiced, 1)
            f0_declination = slope * len(f0_voiced) / (np.mean(f0_voiced) + 1e-10)
        else:
            f0_declination = 0.0
        features.append(float(f0_declination))
        
        # Speaking rate variance (emotions change speaking speed, AI is monotonous)
        try:
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
            if len(onset_frames) > 5:
                onset_intervals = np.diff(onset_frames)
                speaking_rate_var = np.std(onset_intervals) / (np.mean(onset_intervals) + 1e-10)
            else:
                speaking_rate_var = 0.0
        except:
            speaking_rate_var = 0.0
        features.append(float(speaking_rate_var))
        
        # HNR temporal variance (voice quality changes with emotion)
        try:
            n_hnr_windows = min(8, len(y) // sr)
            if n_hnr_windows > 2:
                win_len = len(y) // n_hnr_windows
                hnr_values = []
                for i in range(n_hnr_windows):
                    win_y = y[i*win_len:(i+1)*win_len]
                    win_harmonic = librosa.effects.harmonic(win_y)
                    win_percussive = librosa.effects.percussive(win_y)
                    h_energy = np.sum(win_harmonic**2)
                    p_energy = np.sum(win_percussive**2) + 1e-10
                    hnr_values.append(10 * np.log10(h_energy / p_energy + 1e-10))
                hnr_temporal_var = np.std(hnr_values)
            else:
                hnr_temporal_var = 0.0
        except:
            hnr_temporal_var = 0.0
        features.append(float(hnr_temporal_var))
        
        # Phrase boundary detection (pitch resets at phrase boundaries)
        if f0 is not None and len(f0) > 50:
            f0_diff = np.abs(np.diff(f0))
            pitch_resets = len(find_peaks(f0_diff, height=np.std(f0_diff)*2)[0])
            features.append(float(pitch_resets))
        else:
            features.append(0.0)
        
        # Energy phrase pattern (emotional speech has phrase-level emphasis)
        try:
            long_rms = librosa.feature.rms(y=y, frame_length=4096, hop_length=2048)[0]
            energy_phrase_var = np.std(long_rms) / (np.mean(long_rms) + 1e-10)
            features.append(float(energy_phrase_var))
        except:
            features.append(0.0)
        
        # Intensity-pitch correlation at phrase level
        try:
            if f0 is not None and len(f0[f0 > 0]) > 10:
                # Downsample both to phrase level
                n_phrases = min(10, len(long_rms))
                phrase_len = len(long_rms) // n_phrases
                phrase_energies = [np.mean(long_rms[i*phrase_len:(i+1)*phrase_len]) for i in range(n_phrases)]
                f0_voiced = f0[f0 > 0]
                f0_phrase_len = len(f0_voiced) // n_phrases
                phrase_pitches = [np.mean(f0_voiced[i*f0_phrase_len:(i+1)*f0_phrase_len]) for i in range(n_phrases)]
                phrase_corr = np.corrcoef(phrase_energies, phrase_pitches)[0, 1]
                phrase_corr = phrase_corr if not np.isnan(phrase_corr) else 0.0
            else:
                phrase_corr = 0.0
        except:
            phrase_corr = 0.0
        features.append(float(phrase_corr))
        
        # Creaky voice detection (common at phrase ends, AI lacks this)
        try:
            # Look at final portions of detected segments
            if len(silent_intervals) > 0:
                creaky_count = 0
                for start, end in silent_intervals[-5:]:
                    if end - start > sr // 10:
                        end_portion = y[max(0, end-sr//20):end]
                        if len(end_portion) > 100:
                            # Creaky voice has low-frequency periodicity
                            autocorr = np.correlate(end_portion, end_portion, mode='full')
                            autocorr = autocorr[len(autocorr)//2:]
                            peaks, _ = find_peaks(autocorr[:sr//50], height=np.max(autocorr)*0.3)
                            if len(peaks) > 2:
                                creaky_count += 1
                features.append(float(creaky_count))
            else:
                features.append(0.0)
        except:
            features.append(0.0)
        
        # Total Part 9: 25 (cumulative: 225)
        
        # Pad to 225
        while len(features) < N_FEATURES:
            features.append(0.0)
        
        # Ensure exactly N_FEATURES
        features = features[:N_FEATURES]
        
        # Convert all features to scalars (some librosa functions return arrays)
        scalar_features = []
        for f in features:
            if isinstance(f, np.ndarray):
                scalar_features.append(float(f.flatten()[0]) if f.size > 0 else 0.0)
            else:
                scalar_features.append(float(f))
        
        features_array = np.array(scalar_features, dtype=np.float32)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features_array
    
    except Exception as e:
        raise ValueError(f"Error extracting features from {audio_path}: {str(e)}")


def get_audio_info(audio_path: str) -> dict:
    """Get basic information about an audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        
        if len(y.shape) == 1:
            channels = 1
            duration = len(y) / sr
        else:
            channels = y.shape[0]
            duration = y.shape[1] / sr
        
        return {
            "duration_seconds": round(duration, 2),
            "sample_rate": sr,
            "channels": channels
        }
    except Exception as e:
        raise ValueError(f"Error reading audio file: {str(e)}")


# Feature names for interpretability
FEATURE_NAMES = [
    # Part 1: MFCC & Spectral (40)
    *[f"mfcc_{i}_mean" for i in range(13)],
    *[f"mfcc_{i}_std" for i in range(13)],
    *[f"delta_mfcc_{i}_mean" for i in range(13)],
    "spectral_centroid_mean",
    
    # Part 2: Pitch Analysis (30)
    "pitch_mean", "pitch_std", "pitch_range", "pitch_median",
    "jitter_local", "jitter_rap", "jitter_ppq5", "jitter_ddp", "jitter_std",
    "pitch_roughness", "pitch_inflections", "pitch_range_normalized",
    "pitch_p25", "pitch_p75", "pitch_iqr90",
    "pitch_skew", "pitch_kurtosis", "pitch_entropy",
    "pitch_local_var_mean", "pitch_local_var_std",
    "pitch_accel_mean", "pitch_accel_std",
    "voiced_ratio", "voiced_seg_mean", "voiced_seg_std", "voiced_seg_count",
    "vibrato_strength", "pitch_pad_1", "pitch_pad_2", "pitch_pad_3",
    
    # Part 3: Pause & Breathing (25)
    "energy_mean", "energy_std", "energy_range",
    "pause_count", "pause_mean", "pause_std", "pause_max", "pause_min", "pause_median",
    "pause_entropy", "inter_pause_mean", "inter_pause_std",
    "silence_ratio", "breath_candidates",
    "energy_diff_mean", "energy_diff_std", "energy_accel",
    "attack_mean", "attack_max", "speech_rate", "energy_entropy",
    "pause_pad_1", "pause_pad_2", "pause_pad_3", "pause_pad_4",
    
    # Part 4: Shimmer (20)
    "shimmer_local", "shimmer_db", "shimmer_apq3", "shimmer_apq5", "shimmer_apq11",
    "shimmer_dda", "amplitude_std", "amp_diff_std",
    "zc_interval_mean", "zc_interval_std", "zc_interval_cv",
    "hnr", "nhr", "hp_ratio", "noise_mean", "noise_std",
    "shimmer_pad_1", "shimmer_pad_2", "shimmer_pad_3", "shimmer_pad_4",
    
    # Part 5: Phase & Formants (25)
    "phase_smoothness", "phase_discontinuity", "group_delay_mean", "group_delay_std", "group_delay_skew",
    "inst_freq_std", "inst_freq_diff",
    "spectral_flux_mean", "spectral_flux_std", "spectral_flux_skew",
    "spectral_flatness_mean", "spectral_flatness_std",
    "f1_transition_var", "f2_transition_var", "f1_range", "f2_range",
    "spectral_bandwidth_mean", "spectral_bandwidth_std",
    "spectral_rolloff_mean", "spectral_rolloff_std",
    "phase_pad_1", "phase_pad_2", "phase_pad_3", "phase_pad_4", "phase_pad_5",
    
    # Part 6: Temporal Dynamics (25)
    "mfcc_diff_mean", "mfcc_diff_std", "delta2_mfcc_mean", "delta2_mfcc_std",
    "temporal_flatness", "mod_freq_dominant", "mod_spec_max", "mod_ratio",
    "spectral_contrast_mean", "spectral_contrast_std",
    "long_term_energy_std", "energy_ratio_max_min",
    "tempo", "rhythm_regularity", "onset_count", "onset_interval_std",
    "spectral_entropy_mean", "spectral_entropy_std",
    "temporal_pad_1", "temporal_pad_2", "temporal_pad_3", "temporal_pad_4",
    "temporal_pad_5", "temporal_pad_6", "temporal_pad_7",
    
    # Part 7: Filler Patterns (20)
    "potential_filler_count", "filler_duration_mean", "low_centroid_sustained",
    "speech_rate_var", "hesitation_count", "filler_stability",
    "micro_pause_count", "micro_pause_ratio", "breath_patterns",
    "voiced_energy_std", "voiced_energy_skew", "onset_interval_cv",
    "chroma_var", "chroma_entropy",
    "filler_pad_1", "filler_pad_2", "filler_pad_3", "filler_pad_4",
    "filler_pad_5", "filler_pad_6",
    
    # Part 8: Robustness (15)
    "log_mel_mean", "log_mel_std", "log_mel_diff",
    "mel_temporal_var_mean", "mel_temporal_var_std", "spectral_decay",
    "cepstral_peak", "low_band_ratio", "mid_band_ratio", "high_band_ratio",
    "signal_mean", "signal_std", "signal_kurtosis", "signal_skew",
    "final_pad"
]


def extract_features_from_array(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract features from a numpy audio array (for multi-pass analysis).
    Saves to temp file and processes.
    """
    import tempfile
    import soundfile as sf
    import os
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, y, sr)
    
    try:
        features = extract_features(tmp_path, duration=None)
        return features
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        print(f"Extracting features from: {audio_path}")
        
        info = get_audio_info(audio_path)
        print(f"Audio info: {info}")
        
        features = extract_features(audio_path)
        print(f"Feature vector shape: {features.shape}")
        print(f"\nSample features:")
        for i, name in enumerate(FEATURE_NAMES[:20]):
            print(f"  {name}: {features[i]:.4f}")
    else:
        print("Usage: python features_v3.py <audio_file>")
