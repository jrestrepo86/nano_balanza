function [Z_total] = simulate_Y_4resonators_singleLcoupling_model(x,fexp)
%f = x(1)^2 + a*x(2)^2;




%          /*
%          *
%          *Lm=x[1]
%          *Cm=x[2]
%          *Rm=x[3]
%          *Lk=x[4]
%          *C0=x[5]
%          *Lmass1=x[6]
%          *Lmass2=x[7]
%          *Lmass3=x[8]



w=2*pi*fexp;

Z_Co=1./(1j*w*x(5));% Impedancia ZCo

Z_Q4=(1j*w*x(1))+(1j*w*x(8))+(x(3))+(1./(1j*w*x(2)))+Z_Co;
Z_Q3=(1j*w*x(1))+(1j*w*x(7))+(x(3))+(1./(1j*w*x(2)))+Z_Co;
Z_Q2=(1j*w*x(1))+(1j*w*x(6))+(x(3))+(1./(1j*w*x(2)))+Z_Co;

Z_coupling=(-1j*w*x(4));

Z_half = Z_coupling.*Z_Q4./(Z_coupling+Z_Q4);%paralelo de Z_Q4 con Z_coupling_Lk3
Z_half = Z_half.*Z_Q3./(Z_half+Z_Q3);%paralelo del anterior con ZQ3 
Z_half = Z_half.*Z_coupling./(Z_coupling+Z_half);%paralelo de lo anterior con Z_coupling_Lk2
Z_half = Z_half.*Z_Q2./(Z_half+Z_Q2);%paralelo del anterior con ZQ2 
Z_half = Z_half.*Z_coupling./(Z_coupling+Z_half);%paralelo de lo anterior con Z_coupling_Lk1

%Z_Q1=(1j*w*x(4))+(1j*w*x(1))+(x(3))+(1./(1j*w*x(2)))+Z_half;
Z_Q1=(1j*w*x(1))+(x(3))+(1./(1j*w*x(2)))+Z_half;

Z_total=Z_Co.*Z_Q1./(Z_Co+Z_Q1);%paralelo final






% abs_Zsim=abs(Z_total);
% angle_Zsim=angle(Z_total);






