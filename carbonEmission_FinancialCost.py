import argparse
gpu_dict = {
    'RTX 4090': 450,
    'RTX 4080': 320,
    'RTX 4070 TI': 285,
    'RTX 4070': 200,
    'RTX 4060 TI': 160,
    'RTX 4060': 115,
    'RTX 3090 TI': 450,
    'RTX 3090': 350,
    'RTX 3080 TI': 350,
    'RTX 3080 12GB': 350,
    'RTX 3080 10GB': 320,
    'RTX 3070 TI': 290,
    'RTX 3070': 220,
    'RTX 3060 TI': 200,
    'RTX 3060': 170,
    'RTX 3050': 130,
    'RTX 2080 TI': 250,
    'RTX 2080 Super': 250,
    'RTX 2080': 215,
    'RTX 2070 Super': 215,
    'RTX 2070': 175,
    'RTX 2060 Super': 175,
    'RTX 2060': 160,
    'GTX 1660 TI': 120,
    'GTX 1660 Super': 125,
    'GTX 1660': 120,
    'GTX 1650 Super': 100,
    'GTX 1650': 75,
    'GTX 1080 TI': 250,
    'GTX 1080': 180,
    'GTX 1070 TI': 180,
    'GTX 1070': 150,
    'GTX 1060 6GB': 120,
    'GTX 1060 3GB': 120,
    'GTX 1050 TI': 75,
    'GTX 1050': 75
}

if __name__ == "__main__":
    # valor de gCO2/kwh tirado da média do brasil de 2023 nesse site:
    # https://www.gov.br/mcti/pt-br/acompanhe-o-mcti/noticias/2024/02/fator-de-emissao-de-co2-na-geracao-de-energia-eletrica-no-brasil-em-2023-e-o-menor-em-12-anos
    # usando tarifa média de R$0,67 / KWh

    parser = argparse.ArgumentParser()
    parser.add_argument('--h', type=float, default=3.683)
    parser.add_argument('--gpu', type=str, default='RTX 4070')
    parser.add_argument('--custo_kwh', type=float, default=0.67)
    parser.add_argument('--gCO2', type=float, default=38.5)
    parser.add_argument('--n_gpus', type=int, default=1)

    args = parser.parse_args()
    horas = args.h
    gpu = args.gpu
    custo_kwh = args.custo_kwh
    gCO2_por_KWH = args.gCO2
    numero_gpus = args.n_gpus
    print(args)

    #horas, gpu, numero_gpus = 1, custo_kwh = 0.67, gCO2_por_KWH = 38.5
    watts_gpu = gpu_dict[gpu]
    kwh = horas * watts_gpu * numero_gpus / 1000

    custo = kwh * custo_kwh
    gCO2 = kwh * gCO2_por_KWH

    print('kWh: ', round(kwh, 2), '| Custo: R$', round(custo, 2), ' | gCO2: ', round(gCO2, 2))

    print('Meses de uma árvore para sequestrar o CO2: ',  round(gCO2*0.18/168.96,2), '\n'
                                                                                      'Equivalente de CO2 em quilometros rodados em um carro de passageiro comum: ',
          round(gCO2*0.97/168,2), '\n'
                                       'Equivalente de CO2 em porcentagem de um voo paris-londres: ',
          round(gCO2*0.34/168.96,2), '% ')
