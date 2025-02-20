import os
import pandas as pd

def process_data(data_path: str, min_return: float = 0.05):
    """Process the data"""
    df = pd.read_excel(data_path)
    df = df.iloc[1:]
    df.drop(columns=[
        'Périodicité de cotation', 'Préavis', 'Date création du fonds', 
        'Notation MorningStar', 'Unnamed: 26', 'Indice de risque', 'Date d\'ouverture',
        'Date de fermeture commerciale', 'Date de fin des entrées', 'Éligibilité PEA', 
        'Éligibilité PEA PME', 'Éligibilité Fourgous', 'Unnamed: 27', 'Unnamed: 28',
        'Unnamed: 29', 'Dernière VL 2024', 'Dernière VL 2025', 'Date dernière VL 2025',
        'Actifs net de la part', 'Devise d\'origine du fonds', 'Éligibilité Investissement progressif',
        'Unnamed: 41', 'Éligibilité Sécurisation plus-values', 'Unnamed: 43', 
        'Éligibilité Dynamisation plus-values', 'Unnamed: 45' , 'Éligibilité Stop-loss relatif', 
        'Unnamed: 47', 'Éligibilité Rééquilibrage automatique', 'Unnamed: 49', 'ISR', 'Relance', 
        'Finansol', 'Towards Sustainability', 'ESG Lux Flag', 'Lux Flag Climate Finance',
        'Lux Flag Environment', 'Greenfin', 'ICMA Green Bond Principles', 'FNG Spiegel', 
        'Classification SFDR', 'Investissements durables SFDR', 'Taxonomie (part verte exprimée en %)', 
        'PAI', 'Classification au sens AMF', 'Gestionnaire'
    ], inplace=True)
    df.rename(columns={
        'Code ISIN': 'ISIN', 'Unité de compte': 'Unit of account', 'Performances annuelles': '2018',
        'Unnamed: 16': '2019', 'Unnamed: 17': '2020', 'Unnamed: 18': '2021',
        'Unnamed: 19': '2022', 'Unnamed: 20': '2023', 'Unnamed: 21': '2024',
        'Performances glissantes': '10 years', 'Unnamed: 23': '5 years', 'Unnamed: 24': '3 years',
        'Unnamed: 25': '1 year', 'Volatilités': 'Volatility 5 years',
        'Unnamed: 34': 'Volatility 3 years', 'Unnamed: 35': 'Volatility 1 year',
        'Frais de gestion': 'Fees', 'Ratio de Sharpe': 'Sharpe',
        'Catégorie MorningStar': 'Category',
    }, inplace=True)
    df.fillna(0, inplace=True)
    df['Fees'] = df['Fees'].str.replace(',', '.').str.replace('%', '').astype(float) / 100
    df[df.select_dtypes(include=['float64', 'int64']).columns] = df[df.select_dtypes(include=['float64', 'int64']).columns].round(4)
    df.set_index('ISIN', inplace=True)
    df = df[df[['2020', '2021', '2022', '2023', '2024']].mean(axis=1) > min_return]
    df.to_csv(os.path.join('data', 'processed', 'data.csv'), index=True)

if __name__ == '__main__':
    process_data(os.path.join('data', 'PREVI-OPTIONS.xls'))