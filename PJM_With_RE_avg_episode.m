clc;clear all;
%% Learning Parameter setting
num_epi=3;
myalpha_w=0.1;
myalpha_theta=0.05;
del_t=0.001;
mygamma=1;
sig=1;
Rbar= -1;
mybeta=0.1;

%% RL state initialization
pg=60; pd=62;
So=[pg; pd; 0];
n_train = 5740;
org_state_array=zeros(n_train,3);
final_state_energy = zeros(num_epi,1);
energy = zeros(n_train,3);
reward=zeros(n_train);
final_reward = (n_train);
A_array= zeros(n_train,1);
final_price = zeros(n_train);
mu = zeros(n_train,1);

%% Tile coding initialization
N=8; % num of tilings
p=[20 20 20]; % partition
numTilesInTiling=prod(p);

w= rand(N * prod(p),1);
theta_mu= rand(N * prod(p),1);
theta_sig = rand(N * prod(p),1);

%% Read in Renewable Energy sources from 10/11 -11/8 PJM
solar_file='../data/5_min_solar_gen.csv'; 
wind_file = '../data/15_sec_wind_gen.csv';
solar_data =csvread(solar_file,1,2);
wind_data = csvread(wind_file,1,2);
wind_data_daily = csvread(wind_file,75982,2,[75982 2 81739 2]);


%cat 15 times of complete data section in original file to wind_data
for i=1:15
    wind_data = cat(1,wind_data,wind_data_daily);
end
wind_data_final = zeros(1,ceil(length(wind_data)/20));

% %Since solar power reads every 5 minutes, we transform wind to 5 mins data
%The original file is the newest data on the top, but we want to read the 
%data in the series of time, we read from the bottom
for i=1:ceil(length(wind_data)/20)
    for j=1:20
        wind_data_final(i) = wind_data_final(i)+wind_data(end - (i+j-2));
    end
    i= i+20;
end

%% Read every 15 sec
% wind_data_final = zeros(1,n_train);
% solar_data_final = zeros(1,n_train);
% for i=1:n_train
%     wind_data_final(i) = wind_data(end-i+1);
% end
% for i=1:ceil(n_train/20)
%     for j = 1:20
%         solar_data_final(i) = solar_data(end-(i+j-2));
%     end
%     i=i+20;
% end
    
omg = 0.001*[solar_data(1:n_train)+wind_data(1:n_train)];

%% main
for epi=1:num_epi
   S=So; 
   i=1;
   reach_goal_times=0;
   org_state_array=zeros(n_train,3);
   fprintf("==================Reach goal ! next epi = %d==============================", epi);
   while i<n_train
        feaVec_S=featureState(S,N,p);
        V_oldest = w'*feaVec_S;
        mu(i)=theta_mu' * feaVec_S;
        if mu(i)<0
            mu(i)=0;
        elseif mu(i)>9
            mu(i)=9;
        end
        
        A=action(mu(i),sig);
        A_array(i)=A;
        [S_prime,R] = env(S, A, del_t, omg(i));

        feaVec_Sprime = featureState(S_prime, N,p);
        delta=R  + mygamma*w'*feaVec_Sprime-V_oldest; %- Rbar
        %Rbar=Rbar+ mybeta*delta;
        
        %update w and theta
        w= w + myalpha_w*delta*feaVec_S;
        grad_mu = (A-mu(i))*feaVec_S/sig^2;
        theta_mu = theta_mu + myalpha_theta*delta*grad_mu;
        
        %store state 
        org_state_array(i,1) = S(1);
        org_state_array(i,2) = S(2);
        org_state_array(i,3) = S(3);       

        S=S_prime;
        i=i+1;
        reward(i)=R;

   end
   fprintf("end while\n");
   %Take avg.into consid.
    energy= energy +org_state_array;
    final_reward = final_reward +reward;
    final_price = final_price+A_array;

   if epi == num_epi
        energy=energy/num_epi;
        final_reward = final_reward/num_epi;
        final_price = final_price/num_epi;
   end

    
end

figure()
subplot(2,1,1);
t=1:n_train;
%final_
plot(t,reward);
title('reward');
subplot(2,1,2);
t=1:length(A_array);
plot(t,A_array);
title('price');
xlabel('time(5 min)');

figure()
t=1:length(energy);
plot(t, energy(:,1),'-o','DisplayName', 'Non-RES power supply');
hold on;
plot(t, energy(:,2),'--','DisplayName', 'Power demand');
hold on;
plot(t, energy(:,3), ':', 'DisplayName', 'Energy storage');
hold on;
plot(t, omg(1:length(energy)),'DisplayName', 'RES power supply');
hold on;
plot(t, omg(1:length(energy))+energy(:,1), 'DisplayName', 'RES and non-RES power');

title('Reward = - ( abs(S(3)-1) )');
%title('Reward = - ( 2*abs(S(1)+RE-S(2)) + abs(S(3)-1))');
xlabel('time(5 min)');
ylabel('power(GW) and energy(0.083GWh)');
%legend({'power supply','power demand', 'energy storage', 'RES power',  'RES power + power supply'},'Location','northeast')

%% action selection
function A = action(mu,sig)

    A = mu + randn*sig;
  
    if A<0
        A=0.1;
    end
    if A>10 || isnan(A)
        A=10;
    end
        
    %fprintf('print lambda price = %f', A);
end

%% Environment 
function[S_prime, R]= env(S,A,del_t, RE)
    syms s t
    cd=-0.5; td=0.2; bd=10;
    cg=0.5; tg=0.3; bg=2; k=3;    
   
    S_prime(1) = S(1) + del_t*(1/tg*(A-bg-cg*S(1)-k*S(3)));
    S_prime(2) = S(2) + del_t*1/td*(bd+cd*S(2)-A);
    
    %Set the limit for pd and pg
    if S_prime(1)<0 
        fprintf("S1<0");
        S_prime(1)=0;
    end
    if S_prime(1)>200         
        S_prime(1)=200;
        fprintf("S1>200 --> S1=%d", S_prime(1));
    end
    if S_prime(2)<0 
        fprintf("S2<0");
        S_prime(2)=0;
    end
    if S_prime(2)>200
        S_prime(2)=200;
        fprintf("S2>200 --> S2=%d", S_prime(2));
    end 
    S_prime(3) = del_t*(S_prime(1)-S_prime(2)+RE) + S(3);
    %if S(3)<0
        %R= - ( 2*abs(S(1)+RE-S(2))+abs(S(3)-1)) ;
    %else
        %R= - ( 2*abs(S(1)+RE-S(2)) + abs(S(3)-1)); 
    %end
    R= - ( abs(S(3)-1)) ;


end
%% Tile Coding
function tiles = featureState(S, N, p)
    % N is the number of tilings
    % p is a vector consisting of p_i; p_i is the number of partitions in the i-th dim

    S = state_normal(S); % modify the state normalization if necessary
    numTilesInTiling=prod(p);
    sz=[N p];
    tiles=zeros(N*numTilesInTiling,1);
    L= p./(p-1+1/N);
    xi=(L-1)/(N-1);
    blocklength=L./p;
    
    for n=1:N
        %fprintf("size of xi=%f",size(xi));
        %fprintf("size of (n-1)*xi =%f ", size((n-1)*xi));
        tempS=S+ (n-1)*xi;
        mysub=ceil(tempS./blocklength);
        if mysub(1)==0
            mysub(1)=1;
        end
        if mysub(2)==0
            mysub(2)=1;
        end
        if mysub(3)==0
            mysub(3)=1;
        end
        myind = sub2ind(sz , n, mysub(1),mysub(2),mysub(3)); % for 3 states
        tiles(myind)=1;
    end
end
%% State normalization
function S_norm=state_normal(S)

    S_norm(1) = S(1)/500;
    S_norm(2) = S(2)/500;
    S_norm(3) = (S(3)+500)/1000;
end