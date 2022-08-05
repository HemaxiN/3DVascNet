import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:\iMM\papers\vessel_segmentation\3_04_08_2022\lossescycleGAN.csv', sep=';')

plt.figure()
plt.plot(df['dAlosses1'])
plt.plot(df['dAlosses2'])
plt.legend(['Discriminator A 1', 'Discriminator A 2'])


plt.figure()
plt.plot(df['dBlosses1'])
plt.plot(df['dBlosses2'])
plt.legend(['Discriminator B 1', 'Discriminator B 2'])


plt.figure()
plt.plot(df['glosses1'])
plt.plot(df['glosses2'])
plt.legend(['Generator 1', 'Generator 2'])