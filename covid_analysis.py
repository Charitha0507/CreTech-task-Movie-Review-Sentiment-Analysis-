import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class COVIDAnalyzer:
    def __init__(self):
        self.data = None
        self.global_data = None
        
    def fetch_covid_data(self):
        """Fetch COVID-19 data from Our World in Data"""
        print("📊 Fetching COVID-19 data...")
        
        # Our World in Data COVID-19 dataset URL
        url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
        
        try:
            self.data = pd.read_csv(url)
            print(f"✅ Data loaded successfully! Shape: {self.data.shape}")
            
            # Convert date column
            self.data['date'] = pd.to_datetime(self.data['date'])
            
            # Create global aggregated data
            self.create_global_data()
            
            return True
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return False
    
    def create_global_data(self):
        """Create global aggregated data"""
        # Filter out non-country entries
        countries_only = self.data[~self.data['iso_code'].isin(['OWID_WRL', 'OWID_EUN', 'OWID_INT'])]
        
        # Group by date and sum global cases
        self.global_data = countries_only.groupby('date').agg({
            'new_cases': 'sum',
            'new_deaths': 'sum',
            'total_cases': 'max',
            'total_deaths': 'max'
        }).reset_index()
        
        # Calculate 7-day rolling averages
        self.global_data['new_cases_7day'] = self.global_data['new_cases'].rolling(window=7).mean()
        self.global_data['new_deaths_7day'] = self.global_data['new_deaths'].rolling(window=7).mean()
    
    def basic_info(self):
        """Display basic information about the dataset"""
        print("\n📈 DATASET OVERVIEW")
        print("=" * 50)
        print(f"Total records: {len(self.data):,}")
        print(f"Countries/regions: {self.data['location'].nunique()}")
        print(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        print(f"Total global cases: {self.data['total_cases'].max():,.0f}")
        print(f"Total global deaths: {self.data['total_deaths'].max():,.0f}")
        
        print("\n📊 TOP 10 COUNTRIES BY TOTAL CASES")
        print("-" * 40)
        top_countries = self.data.groupby('location')['total_cases'].max().sort_values(ascending=False).head(10)
        for i, (country, cases) in enumerate(top_countries.items(), 1):
            print(f"{i:2d}. {country:<20} {cases:>12,.0f}")
    
    def plot_global_trends(self):
        """Plot global COVID-19 trends"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily New Cases (Global)', 'Daily New Deaths (Global)', 
                          'Total Cases Over Time', 'Total Deaths Over Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Daily new cases
        fig.add_trace(
            go.Scatter(x=self.global_data['date'], y=self.global_data['new_cases'],
                      mode='lines', name='Daily Cases', opacity=0.3, line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.global_data['date'], y=self.global_data['new_cases_7day'],
                      mode='lines', name='7-day Average', line=dict(color='red', width=3)),
            row=1, col=1
        )
        
        # Daily new deaths
        fig.add_trace(
            go.Scatter(x=self.global_data['date'], y=self.global_data['new_deaths'],
                      mode='lines', name='Daily Deaths', opacity=0.3, line=dict(color='orange')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=self.global_data['date'], y=self.global_data['new_deaths_7day'],
                      mode='lines', name='7-day Average', line=dict(color='darkred', width=3)),
            row=1, col=2
        )
        
        # Total cases
        fig.add_trace(
            go.Scatter(x=self.global_data['date'], y=self.global_data['total_cases'],
                      mode='lines', name='Total Cases', line=dict(color='green', width=2)),
            row=2, col=1
        )
        
        # Total deaths
        fig.add_trace(
            go.Scatter(x=self.global_data['date'], y=self.global_data['total_deaths'],
                      mode='lines', name='Total Deaths', line=dict(color='purple', width=2)),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Global COVID-19 Trends Analysis", showlegend=False)
        fig.show()
    
    def analyze_country_comparison(self, countries=['United States', 'India', 'Brazil', 'Russia', 'France']):
        """Compare COVID-19 trends across countries"""
        country_data = self.data[self.data['location'].isin(countries)].copy()
        
        # Create comparison plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Cases by Country', 'Total Deaths by Country',
                          'Cases per Million', 'Deaths per Million'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set1[:len(countries)]
        
        for i, country in enumerate(countries):
            country_subset = country_data[country_data['location'] == country]
            color = colors[i]
            
            # Total cases
            fig.add_trace(
                go.Scatter(x=country_subset['date'], y=country_subset['total_cases'],
                          mode='lines', name=f'{country} Cases', line=dict(color=color)),
                row=1, col=1
            )
            
            # Total deaths
            fig.add_trace(
                go.Scatter(x=country_subset['date'], y=country_subset['total_deaths'],
                          mode='lines', name=f'{country} Deaths', line=dict(color=color)),
                row=1, col=2
            )
            
            # Cases per million
            fig.add_trace(
                go.Scatter(x=country_subset['date'], y=country_subset['total_cases_per_million'],
                          mode='lines', name=f'{country} Cases/M', line=dict(color=color)),
                row=2, col=1
            )
            
            # Deaths per million
            fig.add_trace(
                go.Scatter(x=country_subset['date'], y=country_subset['total_deaths_per_million'],
                          mode='lines', name=f'{country} Deaths/M', line=dict(color=color)),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Country Comparison Analysis", showlegend=True)
        fig.show()
    
    def correlation_analysis(self):
        """Analyze correlations between different metrics"""
        # Select relevant columns for correlation
        correlation_cols = [
            'total_cases_per_million', 'total_deaths_per_million',
            'new_cases_per_million', 'new_deaths_per_million',
            'total_tests_per_thousand', 'positive_rate',
            'hospital_beds_per_thousand', 'gdp_per_capita',
            'human_development_index', 'life_expectancy'
        ]
        
        # Get latest data for each country
        latest_data = self.data.groupby('location').last().reset_index()
        correlation_data = latest_data[correlation_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_data, dtype=bool))
        sns.heatmap(correlation_data, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, fmt='.2f', square=True, cbar_kws={"shrink": .8})
        plt.title('COVID-19 Metrics Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def time_series_analysis(self):
        """Perform time series analysis to identify patterns"""
        # Focus on global data
        global_ts = self.global_data.set_index('date')
        
        # Calculate growth rates
        global_ts['case_growth_rate'] = global_ts['total_cases'].pct_change() * 100
        global_ts['death_growth_rate'] = global_ts['total_deaths'].pct_change() * 100
        
        # Create seasonal decomposition visualization
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Plot total cases and deaths
        axes[0, 0].plot(global_ts.index, global_ts['total_cases'])
        axes[0, 0].set_title('Global Total Cases')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].plot(global_ts.index, global_ts['total_deaths'], color='red')
        axes[0, 1].set_title('Global Total Deaths')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot daily cases and deaths (7-day average)
        axes[1, 0].plot(global_ts.index, global_ts['new_cases_7day'])
        axes[1, 0].set_title('Daily New Cases (7-day average)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        axes[1, 1].plot(global_ts.index, global_ts['new_deaths_7day'], color='orange')
        axes[1, 1].set_title('Daily New Deaths (7-day average)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Plot growth rates
        axes[2, 0].plot(global_ts.index, global_ts['case_growth_rate'])
        axes[2, 0].set_title('Case Growth Rate (%)')
        axes[2, 0].tick_params(axis='x', rotation=45)
        axes[2, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        axes[2, 1].plot(global_ts.index, global_ts['death_growth_rate'], color='red')
        axes[2, 1].set_title('Death Growth Rate (%)')
        axes[2, 1].tick_params(axis='x', rotation=45)
        axes[2, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def generate_insights(self):
        """Generate key insights from the analysis"""
        print("\n🔍 KEY INSIGHTS")
        print("=" * 50)
        
        # Global statistics
        total_cases = self.global_data['total_cases'].iloc[-1]
        total_deaths = self.global_data['total_deaths'].iloc[-1]
        case_fatality_rate = (total_deaths / total_cases) * 100
        
        print(f"🌍 Global Overview:")
        print(f"   • Total Cases: {total_cases:,.0f}")
        print(f"   • Total Deaths: {total_deaths:,.0f}")
        print(f"   • Case Fatality Rate: {case_fatality_rate:.2f}%")
        
        # Peak analysis
        peak_cases_idx = self.global_data['new_cases_7day'].idxmax()
        peak_cases_date = self.global_data.loc[peak_cases_idx, 'date']
        peak_cases_value = self.global_data.loc[peak_cases_idx, 'new_cases_7day']
        
        peak_deaths_idx = self.global_data['new_deaths_7day'].idxmax()
        peak_deaths_date = self.global_data.loc[peak_deaths_idx, 'date']
        peak_deaths_value = self.global_data.loc[peak_deaths_idx, 'new_deaths_7day']
        
        print(f"\n📈 Peak Analysis:")
        print(f"   • Peak daily cases: {peak_cases_value:,.0f} on {peak_cases_date.strftime('%Y-%m-%d')}")
        print(f"   • Peak daily deaths: {peak_deaths_value:,.0f} on {peak_deaths_date.strftime('%Y-%m-%d')}")
        
        # Recent trends
        recent_data = self.global_data.tail(30)
        recent_case_trend = recent_data['new_cases_7day'].iloc[-1] - recent_data['new_cases_7day'].iloc[0]
        recent_death_trend = recent_data['new_deaths_7day'].iloc[-1] - recent_data['new_deaths_7day'].iloc[0]
        
        print(f"\n📊 Recent Trends (Last 30 days):")
        case_direction = "📈 Increasing" if recent_case_trend > 0 else "📉 Decreasing"
        death_direction = "📈 Increasing" if recent_death_trend > 0 else "📉 Decreasing"
        print(f"   • Cases: {case_direction} by {abs(recent_case_trend):,.0f}")
        print(f"   • Deaths: {death_direction} by {abs(recent_death_trend):,.0f}")
    
    def run_complete_analysis(self):
        """Run the complete COVID-19 analysis"""
        print("🦠 COVID-19 CASE TRENDS ANALYSIS")
        print("=" * 60)
        
        # Fetch data
        if not self.fetch_covid_data():
            print("Failed to fetch data. Please check your internet connection.")
            return
        
        # Basic information
        self.basic_info()
        
        # Generate insights
        self.generate_insights()
        
        print("\n📊 Generating visualizations...")
        
        # Plot global trends
        self.plot_global_trends()
        
        # Country comparison
        self.analyze_country_comparison()
        
        # Correlation analysis
        self.correlation_analysis()
        
        # Time series analysis
        self.time_series_analysis()
        
        print("\n✅ Analysis complete!")

# Main execution
if __name__ == "__main__":
    analyzer = COVIDAnalyzer()
    analyzer.run_complete_analysis()