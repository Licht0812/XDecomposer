import os
file_path = 'autoXRD/spectrum_analysis/__init__.py'
with open(file_path, 'r') as f:
    lines = f.readlines()
new_lines = []
skip = False
found = False
for i, line in enumerate(lines):
    if '@property' in line and i+1 < len(lines) and 'def suspected_mixtures(self):' in lines[i+1]:
        found = True
        skip = True
        new_lines.append('    @property\n')
        new_lines.append('    def suspected_mixtures(self):\n')
        new_lines.append('        spectrum = self.formatted_spectrum\n')
        new_lines.append('        self.model = tf.keras.models.load_model(self.model_path, compile=False)\n')
        new_lines.append('        if not hasattr(self, "fingerprints"):\n')
        new_lines.append('            import numpy as np\n')
        new_lines.append('            self.fingerprints = np.load("reference_fingerprints_full.npy")\n')
        new_lines.append('            self.ref_names = self.reference_phases\n')
        new_lines.append('        prediction_list, confidence_list, scale_list, spec_list = self.reconstruction_enumerate(spectrum)\n')
        new_lines.append('        return prediction_list, confidence_list, [], scale_list, spec_list\n\n')
        new_lines.append('    def reconstruction_enumerate(self, spectrum):\n')
        new_lines.append('        import torch\n')
        new_lines.append('        import torch.nn.functional as F\n')
        new_lines.append('        import numpy as np\n')
        new_lines.append('        current_spectrum = np.array(spectrum)\n')
        new_lines.append('        mixtures = []\n')
        new_lines.append('        confidences = []\n')
        new_lines.append('        scalings = []\n')
        new_lines.append('        spectra = []\n')
        new_lines.append('        sim_threshold = self.min_conf\n')
        new_lines.append('        if sim_threshold > 1.0: sim_threshold /= 100.0\n')
        new_lines.append('        for _ in range(self.max_phases):\n')
        new_lines.append('            x_input = current_spectrum.reshape(1, 3500, 1)\n')
        new_lines.append('            if np.max(x_input) > 0:\n')
        new_lines.append('                x_input = 100.0 * (x_input - np.min(x_input)) / (np.max(x_input) - np.min(x_input) + 1e-8)\n')
        new_lines.append('            pred_pattern = self.model.predict(x_input, verbose=0)[0]\n')
        new_lines.append('            t_pred = torch.from_numpy(pred_pattern).view(1, -1)\n')
        new_lines.append('            t_refs = torch.from_numpy(self.fingerprints)\n')
        new_lines.append('            sims = F.cosine_similarity(t_pred, t_refs)\n')
        new_lines.append('            best_idx = torch.argmax(sims).item()\n')
        new_lines.append('            best_id = self.ref_names[best_idx]\n')
        new_lines.append('            conf = sims[best_idx].item()\n')
        new_lines.append('            if conf < sim_threshold: break\n')
        new_lines.append('            scale = np.dot(current_spectrum, pred_pattern) / (np.dot(pred_pattern, pred_pattern) + 1e-8)\n')
        new_lines.append('            scale = max(0, scale)\n')
        new_lines.append('            mixtures.append(best_id)\n')
        new_lines.append('            confidences.append(conf * 100.0)\n')
        new_lines.append('            scalings.append(scale)\n')
        new_lines.append('            spectra.append(pred_pattern)\n')
        new_lines.append('            current_spectrum = current_spectrum - scale * pred_pattern\n')
        new_lines.append('            current_spectrum = np.maximum(current_spectrum, 0)\n')
        new_lines.append('            if np.max(current_spectrum) < self.cutoff: break\n')
        new_lines.append('        return [mixtures], [confidences], [scalings], [spectra]\n')
        continue
    if skip:
        if ('def ' in line or '@property' in line) and 'suspected_mixtures' not in line:
            skip = False
            new_lines.append(line)
        continue
    new_lines.append(line)
if found:
    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    print("Patch applied successfully")
else:
    print("Could not find suspected_mixtures property")
