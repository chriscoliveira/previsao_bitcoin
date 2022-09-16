# instalar yfinance prophet pandas plotly cython pystan==2.19.1.1 fbprophet openpyxl kaleido

from email import header
import pandas
import yfinance
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.io as pio
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
import warnings
from turtle import color
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
pandas.options.display.float_format = '${:,.2f}'.format

hj = datetime.today().strftime('%Y-%m-%d')

data_ini = '2016-01-01'
print(data_ini, hj)

# baixa os valores do eth
# data_frame_eth = yfinance.download('ETH-USD', data_ini, hj)
# data_frame_eth.tail()

base = pandas.read_excel('vendas.xlsx')


# biblioteca prophet prefisa se uma coluna de indice e outra com a data
# com isso resetamos o index do data frame
# data_frame_eth.reset_index(inplace=True)
# data_frame_eth.tail()


# a biblioteca prophet espera 2 campos um de data e outro numerico
# data se chama 'ds', e o numerico 'y'
# df = data_frame_eth[["Date", "Adj Close"]]
# df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
# df.tail()

df = base[['Data', 'Valor_total']]
df.rename(columns={'Data': 'ds', 'Valor_total': 'y'}, inplace=True)

# grafico de fechamento
fig = go.Figure()
img = fig.add_trace(go.Scatter(x=df['ds'], y=df['y']))
img.show()

# treina o modelo
modelo = Prophet(seasonality_mode='multiplicative')
modelo.fit(df)

# cria um df com datas futuras
df_futuro = modelo.make_future_dataframe(periods=30)

# previsao
previsao = modelo.predict(df_futuro)
# print(previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

prev = plot_plotly(modelo, previsao)
prev.write_image('grafico.jpg')
prev.show()

# remove datas anteriores ao dia de hoje
novo = previsao[~(previsao['ds'] < hj)]


# uni 2 dataframes
xls_previsao = pandas.merge(df, novo, how='outer')
# xls_previsao = xls_previsao[['ds', 'y', 'yhat']]
xls_previsao.rename(
    columns={'ds': 'Data', 'y': 'Valor_venda', 'yhat': 'Previsao'}, inplace=True)

# salva grafico em jpg
venda = xls_previsao.Valor_venda
meta = xls_previsao.Previsao
plt.grid()
venda.plot(color='darkgreen', label='Venda')
meta.plot(color='red', label='Meta')

plt.legend()
plt.savefig('vendas_meta', bbox_inches='tight', transparent=False)

# exporta xlsx
xls_previsao.to_excel('previsao.xlsx', index=False, header=True)
