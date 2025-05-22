import os
import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class ExperimentLogger:
    def __init__(self, parent_dir: Path, scenario_name: str):
        """初始化实验日志记录器，直接在指定的父目录下使用场景名创建实验目录."""
        self.parent_dir = parent_dir
        self.scenario_name = scenario_name
        self.experiment_dir = self.parent_dir / self.scenario_name
        self.start_time = None
        self.log_file = None
        self.results = {}
        
    def start_experiment(self):
        """开始新的实验，创建实验目录"""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = datetime.datetime.now()
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.log_file = self.experiment_dir / "experiment_log.txt"
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"Experiment started at: {self.start_time}\n")
            f.write("="*50 + "\n")
            
    def log_step(self, step_name, message, data=None):
        """记录实验步骤"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"\n[{timestamp}] {step_name}: {message}\n"
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_message)
            
        if data is not None:
            self.results[step_name] = data
            
    def save_results(self, results_dict, filename="results.json"):
        """保存实验结果到JSON文件"""
        results_file = self.experiment_dir / filename
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=4, default=str)
            
    def save_dataframe(self, df, filename):
        """保存DataFrame到CSV文件"""
        df.to_csv(self.experiment_dir / filename)
        
    def save_plot(self, fig, filename):
        """保存图表"""
        fig.savefig(self.experiment_dir / filename)
        plt.close(fig)
        
    def end_experiment(self):
        """结束实验，生成总结报告"""
        end_time = datetime.datetime.now()
        duration = end_time - self.start_time
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "="*50 + "\n")
            f.write(f"Experiment ended at: {end_time}\n")
            f.write(f"Total duration: {duration}\n")
            
        # 生成实验总结
        self._generate_summary()
        
    def _generate_summary(self):
        """生成实验总结报告"""
        summary_file = self.experiment_dir / "experiment_summary.txt"
        
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("Experiment Summary\n")
            f.write("="*50 + "\n\n")
            
            # 基本信息
            f.write("Basic Information:\n")
            f.write(f"Start Time: {self.start_time}\n")
            f.write(f"End Time: {datetime.datetime.now()}\n")
            f.write(f"Duration: {datetime.datetime.now() - self.start_time}\n\n")
            
            # 主要结果
            f.write("Key Results:\n")
            if "DES Optimization" in self.results:
                des_results = self.results["DES Optimization"]
                f.write("\nDES Optimization Results:\n")
                for key, value in des_results.items():
                    f.write(f"  {key}: {value:.2f}\n")
                    
            if "Carbon Price Scenarios" in self.results:
                f.write("\nCarbon Price Scenarios:\n")
                f.write(f"  Number of scenarios: {len(self.results['Carbon Price Scenarios'])}\n")
                
            if "ROA Results" in self.results:
                f.write("\nReal Option Analysis Results (CCS Investment):\n")
                roa_data = self.results["ROA Results"]
                if isinstance(roa_data, dict):
                    for key, value in roa_data.items():
                        try:
                            f.write(f"  {key.replace('_', ' ').capitalize()}: {float(value):,.2f} CNY\n")
                        except (ValueError, TypeError):
                            f.write(f"  {key.replace('_', ' ').capitalize()}: {value}\n")
                else:
                    f.write(f"  Raw Data: {roa_data}\n")
                    
            # 决策结果
            if "Operational Hedging Decision" in self.results:
                f.write("\nOperational Hedging Decision:\n")
                decision_details = self.results["Operational Hedging Decision"]
                if isinstance(decision_details, dict):
                    f.write(f"  Decision: {decision_details.get('decision', 'N/A')}\n")
                    if 'details' in decision_details and isinstance(decision_details['details'], dict):
                        for key, value in decision_details['details'].items():
                            formatted_key = key.replace('_', ' ').capitalize()
                            if isinstance(value, float):
                                f.write(f"    {formatted_key}: {value:.2f}\n")
                            else:
                                f.write(f"    {formatted_key}: {value}\n")
                    else:
                        f.write(f"    Raw Details: {decision_details.get('details')}\n")
                else:
                    f.write(f"  Raw Data: {decision_details}\n")

            if "Strategic CCS Investment Decision" in self.results:
                f.write("\nStrategic CCS Investment Decision:\n")
                decision_details = self.results["Strategic CCS Investment Decision"]
                if isinstance(decision_details, dict):
                    for key, value in decision_details.items():
                        formatted_key = key.replace('_', ' ').capitalize()
                        f.write(f"  {formatted_key}: {value}\n")
                else:
                    f.write(f"  Raw Data: {decision_details}\n")
                
    def get_experiment_dir(self):
        """获取当前实验目录"""
        return str(self.experiment_dir) 