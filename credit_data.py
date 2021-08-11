import pandas as pd

class CreditData:
  def get_data(self, file_name):
    data_path = "data_train_final1.csv"
    data = pd.read_csv(file_name, encoding = "utf-8")
    #print(len(data), data.head())


    # Colunas com correlação mais baixa que atrapalham a performance do modelo
    dropcolumns = [ 'NEGADO ANTERIORMENTE',
      'ATRASO 60 DIAS',
      'CLIENTE EM ATRASO',
      'REGRA - NEGADO ANTERIORMENTE',
      'REGRA - RENOV. COMP. RESIDEN',
      'REGRA - FINANCIAMENTO P/ AUT',
      'REGRA - CLIENTE EM ATRASO',
      'REGRA - REGRA MOVEIS',
      'RESIDENCIA CEDIDA',
      'STATUS COMPROVANTE RESIDENCIA']

    try:
      data = data.drop(dropcolumns, axis = 1)
    except Exception as e:
      print(e)

    #Movendo as colunas:
    data = data[["STATUS2", "COD_FILIAL", "VALOR", "EXP. PROFISSIONAL","PRIMEIRA COMPRA", "CAPACIDADE FINANCIA", "CLI. RESTR. MERCADO",
             "REGRA - COD_FILIAL", "REGRA - IDADE PERMITIDA", "REGRA - EXP. PROFISSIONAL",
             "REGRA - TEMPO DE RESIDENCIA", "REGRA - RENOV. COM. RENDA", "REGRA - COMPRA S/ ENTRADA",
             "REGRA - CLIENTE SCORE X", "REGRA - CAPACIDADE FINANCIA", "REGRA - ATRASO 60 DIAS",
             "REGRA - VERIFICACAO FIADOR", "REGRA - CLI. RESTR. MERCADO", "REGRA - CLIENTE ALTO RISCO",
             "REGRA - COMPRA S/ ENTRADA", "NUMERO DE PARCELAS", "NEGADO A N DIAS", "NEGADO HOJE",
             "RESIDENCIA ALUGADA", "RESIDENCIA PROPRIA", "TIPO RENDA", "STATUS COMPROVANTE RENDA",
             "CLIENTE CLASSIFICACAO"]]

    # Transformando dados pro tipo necessário:
    data.loc[:] = data.loc[:].apply(lambda x: x.astype(float))
    Y = data.iloc[:,0].values
    #Y = [y.astype(int) for y in data.iloc[:,0].values]
    X = data.iloc[:,1:28].values

    return X, Y
