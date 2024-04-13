function [Z_total] = New_simulate_Y_4resonators_singleLcoupling_model(x,fexp)
%f = x(1)^2 + a*x(2)^2;




%          /*
%          *
%          *Lm=x[1]
%          *Cm=x[2]
%          *Rm=x[3]
%          *C0=x[4]
%          *Lk=x[5]
%          *Lmass1=x[6]
%          *Lmass2=x[7]
%          *Lmass3=x[8]



w=2*pi*fexp;

Z_Co=1./(1j*w*x(4));% Impedancia ZCo

Z_Q1=(1j*w*x(1))+(1j*w*x(6))+(x(3))+(1./(1j*w*x(2)))+Z_Co;
Z_Q2=(1j*w*x(1))+(1j*w*x(7))+(x(3))+(1./(1j*w*x(2)))+Z_Co;
Z_Q3=(1j*w*x(1))+(1j*w*x(8))+(x(3))+(1./(1j*w*x(2)))+Z_Co;

%Calculamos las impedancias series y paralelos
Z_coupling=(-1j*w*x(5));

Z_half = Z_coupling + Z_Q3;%serie entre Z_Q3 + Zcoupling_Lk9
Z_half = Z_coupling.*Z_half./(Z_coupling+Z_half);%paralelo de Z_half con Z_coupling_Lk8
Z_half = Z_half + Z_coupling;%serie entre paralelo anterior + Zcoupling_Lk7
Z_half = Z_half.*Z_Q2./(Z_half+Z_Q2);%paralelo de Z_Q2 con Z_half
Z_half = Z_half + Z_coupling;%serie entre paralelo anterior + Zcoupling_Lk6
Z_half = Z_half.*(Z_coupling)./(Z_coupling+Z_half);%paralelo de Z_half con Z_coupling_Lk5
Z_half = Z_half + Z_coupling;%serie entre paralelo anterior + Zcoupling_Lk4
Z_half = Z_half.*Z_Q1./(Z_half+Z_Q1);%paralelo de Z_Q1 con Z_half
Z_half = Z_half + Z_coupling;%serie entre paralelo anterior + Zcoupling_Lk3
Z_half = Z_half.*(Z_coupling)./(Z_coupling+Z_half);%paralelo de Z_half con Z_coupling_Lk2
Z_half = Z_half + Z_coupling;%serie entre paralelo anterior + Zcoupling_Lk1

Z_Q0=(1j*w*x(1))+(1./(1j*w*x(2)))+(x(3))+Z_half;
Z_total=Z_Co.*Z_Q0./(Z_Co+Z_Q0);%paralelo final

% abs_Zsim=abs(Z_total);
% angle_Zsim=angle(Z_total);






