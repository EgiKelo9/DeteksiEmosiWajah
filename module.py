import cv2
import pickle
import numpy as np
from tqdm import tqdm
from cvxopt import matrix, solvers
from collections import defaultdict
from sklearn.metrics import accuracy_score

def convert_to_grayscale(image):
    if len(np.asarray(image).shape) == 2:
        return image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def resize_image(image, size=None):
    if size is None:
        size = (128, 128)
    if image is None or image.size == 0:
        raise ValueError("Empty or invalid image provided to resize_image")
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized_image

def normalize_illumination(image):
    equalized = cv2.equalizeHist(image)
    return equalized

def reduce_noise(image, kernel_size=3):
    denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return denoised

def enhance_features(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 3)
    enhanced = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return enhanced

def preprocess_image(image):
    image = convert_to_grayscale(image)
    image = resize_image(image)
    image = normalize_illumination(image)
    image = reduce_noise(image)
    image = enhance_features(image)
    return image

class CannyFeatureExtractor:
    
    # Fungsi konstruktor untuk menginisialisasi parameter deteksi tepi
    def __init__(self, low_threshold=100, high_threshold=200, aperture_size=3, grid_size=(4,4), visualize=False):
        
        # Inisialisasi parameter untuk ekstraksi fitur
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.aperture_size = aperture_size
        self.grid_size = grid_size
        self.visualize = visualize

    # Fungsi untuk mendeteksi tepi menggunakan metode Canny
    def detect_edges(self, image):
        
        # Pastikan nilai gambar berada dalam rentang 0-255
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
            
        # Terapkan deteksi tepi Canny
        edges = cv2.Canny(image, self.low_threshold, self.high_threshold, apertureSize=self.aperture_size)
        return edges
    
    # Fungsi untuk menghitung histogram arah tepi
    def compute_edge_direction_histogram(self, image, edges):
        
        # Kalkulasi gradien menggunakan Sobel
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        
        # Kalkulasi magnitudo dan arah gradien
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x) % np.pi
        
        # Hanya pertimbangkan arah pada piksel tepi
        edge_directions = direction[edges > 0]
        edge_magnitudes = magnitude[edges > 0]
        
        # Buat histogram dengan 8 bin untuk arah tepi
        hist, _ = np.histogram(edge_directions, bins=8, range=(0, np.pi), weights=edge_magnitudes)
        
        # Normalisasi histogram
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)
            
        return hist
    
    # Fungsi untuk mengekstrak fitur dari gambar menggunakan grid
    def extract_grid_features(self, image, edges):
        
        # Kalkulasi dimensi setiap sel
        height, width = edges.shape
        rows, cols = self.grid_size
        cell_height = height // rows
        cell_width = width // cols
        
        # Inisialisasi list untuk menyimpan fitur
        features = []
        
        # Loop melalui setiap sel dalam grid
        for i in range(rows):
            for j in range(cols):
                
                # Dapatkan koordinat sel
                y_start = i * cell_height
                y_end = min((i + 1) * cell_height, height)
                x_start = j * cell_width
                x_end = min((j + 1) * cell_width, width)
                
                # Ekstrak sel dari citra dan tepi
                cell_image = image[y_start:y_end, x_start:x_end]
                cell_edges = edges[y_start:y_end, x_start:x_end]
                
                # Kalkulasi ciri statistik dasar dari sel
                edge_density = np.sum(cell_edges > 0) / cell_edges.size
                edge_directions = self.compute_edge_direction_histogram(cell_image, cell_edges)
                
                if np.any(cell_edges > 0):
                    # Hitung piksel dalam setiap komponen
                    num_labels, labels = cv2.connectedComponents(cell_edges)
                    component_sizes = [np.sum(labels == i) for i in range(1, num_labels)]
                    if component_sizes:
                        avg_edge_length = np.mean(component_sizes)
                        max_edge_length = np.max(component_sizes)
                        num_edges = len(component_sizes)
                    else:
                        avg_edge_length = 0
                        max_edge_length = 0
                        num_edges = 0
                else:
                    avg_edge_length = 0
                    max_edge_length = 0
                    num_edges = 0
                    
                # Tambahkan seluruh fitur ke list
                cell_features = [edge_density, avg_edge_length, max_edge_length, num_edges]
                cell_features.extend(edge_directions)
                features.extend(cell_features)
        
        return np.array(features)
    
    # Fungsi untuk mengekstrak fitur dari gambar
    def extract_features(self, image):
        
        # Panggil fungsi deteksi tepi dan ekstraksi fitur grid
        edges = self.detect_edges(image)
        features = self.extract_grid_features(image, edges)
        
        # Jika visualize diaktifkan, kembalikan fitur dan tepi
        if self.visualize:
            return features, edges
        return features
    
class GLCMFeatureExtractor:
    
    # Fungsi konstruktor untuk menginisialisasi parameter GLCM
    def __init__(self, gray_levels=8, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normalized=True):
        
        # Inisialisasi parameter untuk ekstraksi fitur GLCM
        self.gray_levels = gray_levels
        self.distances = distances
        self.angles = angles
        self.symmetric = symmetric
        self.normalized = normalized
    
    # Fungsi untuk mengkuantisasi citra menjadi level abu-abu
    def quantize_image(self, image):
        
        # Inisialisasi minimum dan maksimum nilai piksel
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val == min_val:
            return np.zeros_like(image, dtype=np.int32)
        
        # Kuantisasi citra menjadi level abu-abu
        bins = np.linspace(min_val, max_val, self.gray_levels + 1)
        quantized = np.digitize(image, bins) - 1
        quantized[quantized == self.gray_levels] = self.gray_levels - 1
        return quantized
    
    # Fungsi untuk menghitung GLCM
    def compute_glcm(self, quantized_image):
        
        # Dapatkan dimensi citra
        height, width = quantized_image.shape
        
        # Inisialisasi list untuk menyimpan GLCM
        glcms = []
        
        # Loop melalui setiap jarak dan sudut
        for distance in self.distances:
            for angle in self.angles:
                
                # Inisialisasi GLCM
                glcm = np.zeros((self.gray_levels, self.gray_levels), dtype=np.float32)
                
                # Hitung offset berdasarkan jarak dan sudut
                dx = int(round(distance * np.cos(angle)))
                dy = int(round(distance * np.sin(angle)))
                
                # Iterasi melalui setiap piksel dalam citra
                for i in range(height):
                    for j in range(width):
                        
                        # Dapatkan nilai piksel referensi
                        i_ref = i
                        j_ref = j
                        val_ref = quantized_image[i_ref, j_ref]
                        
                        # Dapatkan nilai piksel tetangga
                        i_nbr = i + dy
                        j_nbr = j + dx
                        
                        # Cek apakah tetangga berada dalam batas
                        if 0 <= i_nbr < height and 0 <= j_nbr < width:
                            val_nbr = quantized_image[i_nbr, j_nbr]
                            glcm[val_ref, val_nbr] += 1
                        
                # Buat simetris jika ditentukan
                if self.symmetric:
                    glcm = glcm + glcm.T
                    
                # Normalisasi jika ditentukan
                if self.normalized:
                    if np.sum(glcm) > 0:
                        glcm = glcm / np.sum(glcm)
                
                # Tambahkan GLCM ke list
                glcms.append(glcm)

        return glcms
    
    # Fungsi untuk menghitung fitur dari GLCM
    def compute_glcm_features(self, glcm):
        
        # Inisialisasi dimensi dan dictionary untuk menyimpan fitur
        features = {}
        rows, cols = np.indices(glcm.shape)
        
        if np.sum(glcm) > 0:
            
            # Hitung Contrast
            features['contrast'] = np.sum(glcm * ((rows - cols) ** 2))
            
            # Hitung Dissimilarity
            features['dissimilarity'] = np.sum(glcm * np.abs(rows - cols))
            
            # Hitung Homogeneity
            features['homogeneity'] = np.sum(glcm / (1 + (rows - cols) ** 2))
            
            # Hitung Energy / Angular Second Moment
            features['energy'] = np.sum(glcm ** 2)
            
            # Hitung Correlation
            mu_i = np.sum(rows * glcm)
            mu_j = np.sum(cols * glcm)
            sigma_i = np.sqrt(np.sum(glcm * (rows - mu_i) ** 2))
            sigma_j = np.sqrt(np.sum(glcm * (cols - mu_j) ** 2))
            if sigma_i > 0 and sigma_j > 0:
                correlation = np.sum(glcm * (rows - mu_i) * (cols - mu_j) / (sigma_i * sigma_j))
                features['correlation'] = correlation
            else:
                features['correlation'] = 0
                
            # Hitung Entropy
            non_zero_glcm = glcm.copy()
            non_zero_glcm[non_zero_glcm == 0] = 1e-10
            features['entropy'] = -np.sum(glcm * np.log2(non_zero_glcm))
            
        else:
            for feature_name in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'entropy']:
                features[feature_name] = 0
        return features
    
    # Fungsi untuk mengekstrak fitur dari citra
    def extract_features(self, image):
        
        # Kuantisasi citra menjadi level abu-abu
        quantized_image = self.quantize_image(image)
        
        # Hitung GLCM
        glcms = self.compute_glcm(quantized_image)
        
        # Hitung nilai fitur untuk setiap GLCM
        all_features = []
        for glcm in glcms:
            features = self.compute_glcm_features(glcm)
            all_features.append(features)
        
        return all_features
    
class SVM_Scratch:
    
    # Fungsi konstruktor untuk menginisialisasi parameter SVM
    def __init__(self, kernel='rbf', C=10.0, degree=3, gamma=0.01, coef0=0.0):
        self.kernel_type = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.models = {}
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = None
        self.is_multiclass = False
        self.kernel = self._get_kernel_function(kernel)
    
    # Fungsi untuk mendapatkan parameter SVM
    def get_params(self, deep=True):
        return {
            "C": self.C,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "degree": self.degree,
            "coef0": self.coef0,
        }

    # Fungsi untuk mengatur parameter SVM
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    # Fungsi untuk menghitung kernel linear
    def _kernel_linear(self, x, y):
        return np.dot(x, y)
    
    # Fungsi untuk menghitung kernel polinomial
    def _kernel_poly(self, x, y):
        return (1 + self.gamma * np.dot(x, y)) ** self.degree

    # Fungsi untuk menghitung kernel RBF
    def _kernel_rbf(self, x, y):
        return np.exp(-self._gamma * np.linalg.norm(x - y) ** 2)

    # Fungsi untuk mendapatkan fungsi kernel berdasarkan jenis kernel
    def _get_kernel_function(self, kernel):
        if kernel == 'linear':
            return self._kernel_linear
        elif kernel == 'poly':
            self.gamma = self.gamma if self.gamma is not None else 1.0
            return self._kernel_poly
        elif kernel == 'rbf':
            return self._kernel_rbf
        else:
            raise ValueError("Unknown kernel")

    # Fungsi untuk melakukan proses fitting model SVM
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.is_multiclass = n_classes > 2

        if self.is_multiclass:
            self.models = {}
            print(f"Training {n_classes} OvR SVM models...\n")
            for cls in tqdm(self.classes, desc="OvR SVM Training"):
                y_binary = np.where(y == cls, 1, -1)
                model = SVM_Scratch(kernel=self.kernel_type, C=self.C, degree=self.degree, gamma=self.gamma)
                model.fit(X, y_binary)
                self.models[cls] = model
        else:
            y = y.astype(float)
            n_samples, n_features = X.shape
            self.X = X
            self.y = y

            if self.kernel_type == 'rbf':
                self._gamma = self.gamma if self.gamma else 1 / n_features

            # Gram matrix
            K = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    K[i, j] = self.kernel(X[i], X[j])

            P = matrix(np.outer(y, y) * K)
            q = matrix(-np.ones(n_samples))
            A = matrix(y.reshape(1, -1))
            b = matrix(0.0)

            if self.C is None:
                G = matrix(-np.eye(n_samples))
                h = matrix(np.zeros(n_samples))
            else:
                G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
                h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))

            solvers.options['show_progress'] = False
            solution = solvers.qp(P, q, G, h, A, b)
            alphas = np.ravel(solution['x'])

            # Support vectors
            sv = alphas > 1e-5
            self.alphas = alphas[sv]
            self.support_vectors = X[sv]
            self.support_vector_labels = y[sv]

            # Intercept
            self.b = np.mean([
                y_k - np.sum(self.alphas * self.support_vector_labels *
                             [self.kernel(x_k, x_i) for x_i in self.support_vectors])
                for (x_k, y_k) in zip(self.support_vectors, self.support_vector_labels)
            ])

    # Fungsi untuk menghitung nilai keputusan
    def project(self, X):
        if self.is_multiclass:
            if isinstance(next(iter(self.models.values())), dict):
                # Convert dictionary models to SVM_Scratch instances
                for cls in self.models:
                    model_dict = self.models[cls]
                    model = SVM_Scratch(kernel=model_dict['kernel_type'])
                    model._load_from_dict(model_dict)
                    self.models[cls] = model
            
            decision_values = np.column_stack([
                model.project(X) for model in self.models.values()
            ])
            return decision_values
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for alpha, sv_y, sv in zip(self.alphas, self.support_vector_labels, self.support_vectors):
                    s += alpha * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b
    
    # Fungsi untuk menghitung probabilitas prediksi
    def predict_proba(self, X):
        if not self.is_multiclass:
            raise NotImplementedError("predict_proba hanya didukung untuk mode multiclass OvR")

        decision = self.project(X)
        # Softmax untuk setiap baris
        exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))  # stabilisasi
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    # Fungsi untuk melakukan prediksi
    def predict(self, X):
        if self.is_multiclass:
            decision = self.project(X)
            predictions = np.argmax(decision, axis=1)
            return int(self.classes[predictions])
        else:
            return np.sign(self.project(X))
        
    # Fungsi untuk menyimpan model ke file
    def save_model(self, filename):
        # Menyimpan atribut model yang diperlukan untuk prediksi
        model_data = {
            'kernel_type': self.kernel_type,
            'C': self.C,
            'degree': self.degree,
            'gamma': self.gamma,
            'coef0': self.coef0,
            'alphas': self.alphas,
            'support_vectors': self.support_vectors,
            'support_vector_labels': self.support_vector_labels,
            'b': self.b,
            'is_multiclass': self.is_multiclass,
            'classes': self.classes if hasattr(self, 'classes') else None
        }
        
        # Jika model multiclass, simpan semua submodel
        if self.is_multiclass:
            model_data['models'] = {cls: model.save_model_dict() for cls, model in self.models.items()}
        
        # Simpan model ke file menggunakan pickle
        with open(filename, 'wb') as file:
            pickle.dump(model_data, file)
        
        print(f"Model berhasil disimpan ke {filename}")
    
    # Add the save_model_dict helper method
    def save_model_dict(self):
        return {
            'kernel_type': self.kernel_type,
            'C': self.C,
            'degree': self.degree,
            'gamma': self.gamma,
            'coef0': self.coef0,
            'alphas': self.alphas,
            'support_vectors': self.support_vectors,
            'support_vector_labels': self.support_vector_labels,
            'b': self.b,
            'is_multiclass': self.is_multiclass
        }
    
    # Add the load_model class method
    @classmethod
    def load_model(cls, filename):
        # Baca file model
        with open(filename, 'rb') as file:
            model_data = pickle.load(file)
        
        # Inisialisasi model kosong
        model = cls(
            kernel=model_data['kernel_type'],
            C=model_data['C'],
            degree=model_data['degree'],
            gamma=model_data['gamma'],
            coef0=model_data['coef0']
        )
        
        # Atur atribut model
        model.alphas = model_data['alphas']
        model.support_vectors = model_data['support_vectors']
        model.support_vector_labels = model_data['support_vector_labels']
        model.b = model_data['b']
        model.is_multiclass = model_data['is_multiclass']
        
        if model_data['classes'] is not None:
            model.classes = model_data['classes']
        
        # Jika model multiclass, muat semua submodel
        if model.is_multiclass:
            model.models = {}
            for cls, submodel_data in model_data['models'].items():
                submodel = cls()
                submodel._load_from_dict(submodel_data)
                model.models[cls] = submodel
                
        model.kernel = model._get_kernel_function(model.kernel_type)
        
        print(f"Model berhasil dimuat dari {filename}")
        return model
    
    # Add the _load_from_dict helper method
    def _load_from_dict(self, model_dict):
        """Memuat atribut model dari dictionary."""
        for key, value in model_dict.items():
            setattr(self, key, value)
        self.kernel = self._get_kernel_function(self.kernel_type)