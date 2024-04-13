%Limpiamos todas las variables del Workspace
clear all;
%close all;
clc;
N=20000;
Masa_min=64.1394e-9; Masa_max=30*64.1394e-9;
%Masa_min=0; Masa_max=30;
Masas=unifrnd(0, Masa_max,3,N);
M=Masas;
%parametros generales de la simulación%
n_points=20000;%número de puntos
%f_ini=4.96e6
%f_final=4.98e6
f_ini=4.965e6;%frecuencia inicial
f_final=4.975e6;%frecuencia final
%f_ini=4.99e6;%frecuencia inicial
%f_final=5.01e6;%frecuencia final
%Definición de los parámetros del resonador (5MHz)
Lm=64.1394e-3;
Cm=16.0371e-15;
Rm=11.42;
C0=43.3903e-12;
Features=[];
% 3 masas
for n=1:N
    Lmass1=Masas(1,n);
    Lmass2=Masas(2,n);
    Lmass3=Masas(3,n);
    %Lmass1=0.1e-3;
    %Lmass2=0.2e-3;
    %Lmass3=0.3e-3;
    coupling=linspace(0.001,0.005,5);
    %figure(2);
    %for i = 1:5
    %Actualizamos el valor de acoplamiento Lk
    Lk=coupling(2)*Lm;
    parameters=[Lm,Cm,Rm,C0,Lk,Lmass1,Lmass2,Lmass3];
    fsim=linspace(f_ini,f_final,n_points);
    [Zsim] = New_simulate_Y_4resonators_singleLcoupling_model(parameters,fsim);
    Ysim=1./Zsim;
    Gsim=real(Ysim);
    Delta_f=fsim(2)-fsim(1);
   % Bsim=imag(Ysim);
   %%%%%%
  G(:,n)=abs(Ysim)';
%end;  figure ;plot(fsim, G)
   %%%%%%
   g=abs(Ysim)';
   [p , x, ancho ]=findpeaks(g, 'MinPeakProminence', 1e-3);
 
   Num_picos=length(x);
   t=-4:5; ti=linspace(-4,5,10000);
   for i=1:Num_picos  
        y=interp1(t, g(x(i)+t), ti, 'spline');
        Fi=fsim(x(i))+ti*(Delta_f);
        [Amp_max, ind_max]=max(y);
        F_max=Fi(ind_max);
      %  figure(i); plot(fsim, g,'.'); hold on; plot(Fi, y, 'r'); plot(F_max, Amp_max, 'ko')
        Features(n,i*3-2:i*3)=[Amp_max, F_max, ancho(i)];
        
   end
end
Fe=Features';
M=Masas;
L=length(M);
% partición 
r=normrnd(0,1,1,L);
[~, orden]=sort(r);
N_train=round(.85*L)
N_test=round(.15*L);
%N_val=round(.15*L);
Input_Train=Fe(:, 1:N_train);
Output_Train=M(:, 1:N_train);
Input_Test=Fe(:, N_train+1:N_train+N_test);
Output_Test=M(:,  N_train+1:N_train+N_test);
%Input_Val=Fe(:,  N_train+N_test+1:N_train+N_test+N_val);
%Output_Val=M(:,   N_train+N_test+1:N_train+N_test+N_val);
%save("datos.mat", "Fe","M", "Input_Train", "Output_Train", "Input_Test", "Input_Val","Output_Val","Output_Test" )
%figure ;plotmatrix(Fe)
%save("datos.mat", "Fe","M", "Input_Train", "Output_Train", "Input_Test","Output_Test" )
save("datos_feli_20k.mat", "G", "Fe","M")

