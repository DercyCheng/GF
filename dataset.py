import numpy as np
import pandas as pd
import pywt
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split

class SoilDataProcessor:
    """Class for loading and processing soil spectral data"""
    
    def __init__(self):
        # Define column mappings
        self.soil_nutrients = {
            'PH': 'PH',
            '有机质(g/kg)': 'OM',
            '全氮(g/kg)': 'TN',
            '全磷(g/kg)': 'TP',
            '全钾(g/kg)': 'TK',
            '速效N(mg/kg)': 'AN',
            '速效p(mg/kg)': 'AP',
            '速效k(mg/kg)': 'AK',
            'B(mg/kg)': 'B',
            'Cu(mg/kg)': 'Cu',
            'Zn(mg/kg)': 'Zn',
            'Fe(mg/kg)': 'Fe',
            'Ca(mg/kg)': 'Ca',
            'Mg(mg/kg)': 'Mg',
            '全碳(g/kg)': 'TC',
            '易氧化有机碳(mg/g)': 'EOC',
            '有机碳含量(g/kg)': 'SOC',
            '水溶性有机碳(mg/g)': 'WOC'
        }
        
        self.environment_info = {
            '海拔测量': 'ELEV',
            'Longitude': 'LONG',
            'latitude': 'LAT',
            '坡度': 'SLOPE',
            '坡向': 'ASPECT',
            '海拔': 'ALT',
            '大于10度积温': 'GDD10',
            '年均降雨': 'AN_RAIN',
            '年均温度': 'AN_TEMP',
            '代数': 'GEN',
            '林龄': 'FOREST_AGE'
        }
        
        self.target_columns = {
            '有机质(g/kg)': 'OM',
            '全碳(g/kg)': 'TC',
            '易氧化有机碳(mg/g)': 'EOC',
            '有机碳含量(g/kg)': 'SOC',
            '水溶性有机碳(mg/g)': 'WOC'
        }
        
        # Define spectral bands
        self.spectral_bands = [f'{i}' for i in range(400, 2501)]
    
    def load_data(self, filepath):
        """Load data from Excel file and normalize column names"""
        try:
            data = pd.read_excel(filepath)
            print(f"Data loaded successfully from {filepath}!")
            
            # Convert column names to string and rename
            data.columns = data.columns.map(lambda x: f'{x}' if isinstance(x, int) else x)
            data.rename(columns={**self.soil_nutrients, **self.environment_info}, inplace=True)
            
            return data
        except Exception as e:
            print(f"Failed to load data from {filepath}: {e}")
            return None
    
    def split_data(self, data, test_size=0.2, random_state=42):
        """Split data into training and test sets"""
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
        return train_data, test_data
    
    def apply_sgd(self, train_data, test_data, bands, window_length=5, polyorder=2):
        """Apply Savitzky-Golay filter and first derivative"""
        # Process training data
        train_sgd_filtered = savgol_filter(train_data[bands], window_length=window_length, polyorder=polyorder, deriv=0, axis=0)
        train_sgd_derivative = savgol_filter(train_sgd_filtered, window_length=window_length, polyorder=polyorder, deriv=1, axis=0)
        
        # Process test data with same parameters
        test_sgd_filtered = savgol_filter(test_data[bands], window_length=window_length, polyorder=polyorder, deriv=0, axis=0)
        test_sgd_derivative = savgol_filter(test_sgd_filtered, window_length=window_length, polyorder=polyorder, deriv=1, axis=0)
        
        return pd.DataFrame(train_sgd_derivative, columns=bands), pd.DataFrame(test_sgd_derivative, columns=bands)
    
    def apply_snv(self, train_data, test_data, bands):
        """Apply Standard Normal Variate (SNV) normalization"""
        # Calculate mean and std from training data only
        train_mean = train_data[bands].mean(axis=0)
        train_std = train_data[bands].std(axis=0)
        
        # Apply to training data
        train_snv = (train_data[bands] - train_mean) / train_std
        
        # Apply same transformation to test data (using training parameters)
        test_snv = (test_data[bands] - train_mean) / train_std
        
        return train_snv, test_snv
    
    def apply_msc(self, train_data, test_data, bands):
        """Apply Multiplicative Scatter Correction (MSC)"""
        # Calculate mean spectrum from training data only
        mean_spectrum = np.mean(train_data[bands], axis=0)
        
        # Apply MSC to training data
        train_msc = np.zeros_like(train_data[bands])
        for i in range(len(train_data)):
            fit = np.polyfit(mean_spectrum, train_data[bands].iloc[i], 1)
            train_msc[i] = (train_data[bands].iloc[i] - fit[1]) / fit[0]
        
        # Apply MSC to test data using same reference spectrum
        test_msc = np.zeros_like(test_data[bands])
        for i in range(len(test_data)):
            fit = np.polyfit(mean_spectrum, test_data[bands].iloc[i], 1)
            test_msc[i] = (test_data[bands].iloc[i] - fit[1]) / fit[0]
        
        return pd.DataFrame(train_msc, columns=bands), pd.DataFrame(test_msc, columns=bands)
    
    def apply_dwt(self, train_data, test_data, bands, wavelet='db4', level=3):
        """Apply Discrete Wavelet Transform (DWT)"""
        # Apply DWT to training data
        train_dwt = np.zeros((len(train_data), len(bands)))
        for i in range(len(train_data)):
            coeffs = pywt.wavedec(train_data[bands].iloc[i], wavelet, level=level)
            rec = pywt.waverec(coeffs, wavelet)
            if len(rec) > len(bands):
                rec = rec[:len(bands)]
            elif len(rec) < len(bands):
                rec = np.pad(rec, (0, len(bands) - len(rec)))
            train_dwt[i] = rec
        
        # Apply DWT to test data
        test_dwt = np.zeros((len(test_data), len(bands)))
        for i in range(len(test_data)):
            coeffs = pywt.wavedec(test_data[bands].iloc[i], wavelet, level=level)
            rec = pywt.waverec(coeffs, wavelet)
            if len(rec) > len(bands):
                rec = rec[:len(bands)]
            elif len(rec) < len(bands):
                rec = np.pad(rec, (0, len(bands) - len(rec)))
            test_dwt[i] = rec
        
        return pd.DataFrame(train_dwt, columns=bands), pd.DataFrame(test_dwt, columns=bands)
    
    def create_dataset_pairs(self, train_data, test_data):
        """Create all dataset pairs with different preprocessing methods"""
        dataset_pairs = {}
        
        # Get references to column groups
        soil_nutrient_cols = list(self.soil_nutrients.values())
        env_info_cols = list(self.environment_info.values())
        target_cols = list(self.target_columns.values())
        
        # Process spectral bands with different methods
        train_sgd, test_sgd = self.apply_sgd(train_data, test_data, self.spectral_bands)
        train_snv, test_snv = self.apply_snv(train_data, test_data, self.spectral_bands)
        train_msc, test_msc = self.apply_msc(train_data, test_data, self.spectral_bands)
        train_dwt, test_dwt = self.apply_dwt(train_data, test_data, self.spectral_bands)
        train_msc_dwt, test_msc_dwt = self.apply_dwt(
            pd.concat([train_msc], axis=1), 
            pd.concat([test_msc], axis=1), 
            self.spectral_bands
        )
        
        # Create different dataset combinations
        
        # 1. Raw spectral bands with different preprocessing
        dataset_pairs["SBSD"] = (
            pd.concat([train_data[target_cols], train_sgd], axis=1),
            pd.concat([test_data[target_cols], test_sgd], axis=1)
        )
        
        dataset_pairs["SBSNV"] = (
            pd.concat([train_data[target_cols], train_snv], axis=1),
            pd.concat([test_data[target_cols], test_snv], axis=1)
        )
        
        dataset_pairs["SBMSC"] = (
            pd.concat([train_data[target_cols], train_msc], axis=1),
            pd.concat([test_data[target_cols], test_msc], axis=1)
        )
        
        dataset_pairs["SBDWT"] = (
            pd.concat([train_data[target_cols], train_dwt], axis=1),
            pd.concat([test_data[target_cols], test_dwt], axis=1)
        )
        
        # 2. Soil nutrients + spectral bands
        dataset_pairs["SNSB"] = (
            pd.concat([train_data[soil_nutrient_cols], train_data[self.spectral_bands]], axis=1),
            pd.concat([test_data[soil_nutrient_cols], test_data[self.spectral_bands]], axis=1)
        )
        
        dataset_pairs["SNSBSD"] = (
            pd.concat([train_data[soil_nutrient_cols], train_sgd], axis=1),
            pd.concat([test_data[soil_nutrient_cols], test_sgd], axis=1)
        )
        
        dataset_pairs["SNSBSNV"] = (
            pd.concat([train_data[soil_nutrient_cols], train_snv], axis=1),
            pd.concat([test_data[soil_nutrient_cols], test_snv], axis=1)
        )
        
        dataset_pairs["SNSBMSC"] = (
            pd.concat([train_data[soil_nutrient_cols], train_msc], axis=1),
            pd.concat([test_data[soil_nutrient_cols], test_msc], axis=1)
        )
        
        dataset_pairs["SNSBDWT"] = (
            pd.concat([train_data[soil_nutrient_cols], train_dwt], axis=1),
            pd.concat([test_data[soil_nutrient_cols], test_dwt], axis=1)
        )
        
        dataset_pairs["SNSBMSC-DWT"] = (
            pd.concat([train_data[soil_nutrient_cols], train_msc_dwt], axis=1),
            pd.concat([test_data[soil_nutrient_cols], test_msc_dwt], axis=1)
        )
        
        # 3. Soil nutrients + spectral bands + environmental info
        dataset_pairs["SNSBE"] = (
            pd.concat([train_data[soil_nutrient_cols], train_data[self.spectral_bands], train_data[env_info_cols]], axis=1),
            pd.concat([test_data[soil_nutrient_cols], test_data[self.spectral_bands], test_data[env_info_cols]], axis=1)
        )
        
        dataset_pairs["SNSBESD"] = (
            pd.concat([train_data[soil_nutrient_cols], train_sgd, train_data[env_info_cols]], axis=1),
            pd.concat([test_data[soil_nutrient_cols], test_sgd, test_data[env_info_cols]], axis=1)
        )
        
        dataset_pairs["SNSBESNV"] = (
            pd.concat([train_data[soil_nutrient_cols], train_snv, train_data[env_info_cols]], axis=1),
            pd.concat([test_data[soil_nutrient_cols], test_snv, test_data[env_info_cols]], axis=1)
        )
        
        dataset_pairs["SNSBEMSC"] = (
            pd.concat([train_data[soil_nutrient_cols], train_msc, train_data[env_info_cols]], axis=1),
            pd.concat([test_data[soil_nutrient_cols], test_msc, test_data[env_info_cols]], axis=1)
        )
        
        dataset_pairs["SNSBEDWT"] = (
            pd.concat([train_data[soil_nutrient_cols], train_dwt, train_data[env_info_cols]], axis=1),
            pd.concat([test_data[soil_nutrient_cols], test_dwt, test_data[env_info_cols]], axis=1)
        )
        
        dataset_pairs["SNSBEMSC-DWT"] = (
            pd.concat([train_data[soil_nutrient_cols], train_msc_dwt, train_data[env_info_cols]], axis=1),
            pd.concat([test_data[soil_nutrient_cols], test_msc_dwt, test_data[env_info_cols]], axis=1)
        )
        
        return dataset_pairs
    
    def calculate_statistics(self, data, targets, sample_type=""):
        """Calculate statistics for target columns"""
        stats = []
        for col in targets:
            values = data[col].values
            sample_size = len(values)
            min_val = np.min(values)
            max_val = np.max(values)
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv_val = (std_val / mean_val) * 100 if mean_val != 0 else 0
            
            stats.append({
                "样本类型": sample_type,
                "指标": col,
                "样本量": sample_size,
                "最小值": min_val,
                "最大值": max_val,
                "平均值": mean_val,
                "标准差": std_val,
                "变异系数(%)": cv_val
            })
        return pd.DataFrame(stats)
    
    def load_and_process_data(self, filepath, test_size=0.2, random_state=42):
        """Main function to load and process all data"""
        # Load data
        data = self.load_data(filepath)
        if data is None:
            return None
        
        # Split data properly
        train_data, test_data = self.split_data(data, test_size, random_state)
        
        # Create all dataset pairs
        dataset_pairs = self.create_dataset_pairs(train_data, test_data)
        
        # Calculate statistics
        target_cols = list(self.target_columns.values())
        statistics = {
            'all': self.calculate_statistics(data, target_cols, "全部样本"),
            'train': self.calculate_statistics(train_data, target_cols, "训练集"),
            'test': self.calculate_statistics(test_data, target_cols, "测试集")
        }
        
        return {
            'dataset_pairs': dataset_pairs,
            'statistics': statistics,
            'target_columns': target_cols
        }
