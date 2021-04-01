x1=[0.4329 0.3024 0.1349 0.3374 1.1434 1.3749 0.7221 0.4403 -0.5231 0.3255 0.5824 0.1340 0.1480 0.7359 0.7115 0.8251 0.1569 0.0033 0.4243 1.0490 1.4276 0.5971 0.8475 1.3967 0.0044 0.2201 0.6300 -0.2479 -0.3088 -0.5180 0.6833 0.4353 -0.1069 0.4662 0.8298];
x2=[-1.3719 0.2286 -0.6445 -1.7163 -0.0485 -0.5071 -0.7587 -0.8072 0.3548 -2 1.3915 0.6081 -0.2988 0.1869 -1.1469 -1.2840 0.3712 0.6835 0.8313 0.1326 0.5331 1.4865 2.1479 -0.4171 1.5378 -0.5668 -1.2480 0.8960 -0.0929 1.4974 0.8266 -1.4066 -3.2329 0.6261 -1.4089];
x3=[0.7022 0.8630 1.0530 0.3670 0.6637 0.4464 0.7681 0.5154 0.2538 0.7112 -0.2291 0.4450 0.4778 -0.0872 0.3394 0.8452 0.8825 0.5389 0.2634 0.9138 -0.0145 0.2904 0.3179 0.6443 0.6099 0.0515 0.8591 0.0547 0.8659 0.5453 0.0829 0.4207 0.1856 0.7304 0.33119 ];
x4=[-0.8535 2.7909 0.5687 -0.6283 1.2606 1.3009 -0.5592 -0.3129 1.5776 -1.1209 4.1735 3.2230 0.8649 2.3584 0.9573 1.2382 1.7633 2.8249 3.5855 1.9792 3.7286 4.6069 5.8235 1.3927 4.7755 0.7829 0.8093 1.7381 1.5483 2.3993 2.8864 -0.4879 -2.4572 3.4370 1.3235 ];
x5=[-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1];  %threshold matrix
target=[1 -1 -1 -1 1 1 1 1 -1 1 -1 -1 1 1 -1 -1 1 -1 -1 1 1 -1 -1 1 -1 1 -1 1 -1 1 1 1 -1 -1 -1];
samples=35;          %input data 
 %  Training inital weights T(1,2,3,4,5)
 %if you want to test each Training just comment and uncoment the following
 %weights same as #IF DEFINED IN C
           
w1=0.8057;w2=0.6469;w3=0.3897;w4=0.3658;b=0.9004;            
%w1=0.2647;w2=0.8049;w3=0.4430;w4=0.0938;b= 0.9150;                                                                                                                                                                                            
%w1=0.6006;w2=0.7162;w3=0.1565;w4=0.8315;b=0.5844;
%w1=0.9190;w2=0.3115;w3=0.9286;w4=0.6983;b=0.8680;
%w1=0.3575;w2=0.5155;w3=0.4863;w4=0.2155;b=0.3110;

%% 
%network paramaters 

alpha=0.0025;
epoch_limit=3000;
precision=10^-6;
e=0;
E=zeros(1000,1); % memory allocation To save current values of error with epoches 
current_error=0;
delw1=0;delw2=0;delw3=0;delw4=0;delb=0;
epoch=0;
ebison=2;
temp=0;
while( epoch<=epoch_limit && ebison>=precision)  %epoch loop get out 1) epoch>=3000 limit or ebison=10^-6
    
    epoch=epoch+1; % epoch counter
    e=0;
    for i=1:35  %  loop for each sample ,35 loops is one epoch
        u=w1*x1(i)+w2*x2(i)+w3*x3(i)+w4*x4(i)-b;       
        e=(target(i)-u)^2; %calculate the MSE E
        current_error=current_error+e; % acumlate MSE of entire sample
        delw1=alpha*(target(i)-u)*x1(i); % %Here is the difference between perceptron and adaline delta is (d-u) not (d-y) y is fired 
        delw2=alpha*(target(i)-u)*x2(i);
        delw3=alpha*(target(i)-u)*x3(i);
        delw4=alpha*(target(i)-u)*x4(i);
        delb=alpha*(target(i)-u)*x5(i);
        w1=w1+delw1;   % new weights 
        w2=w2+delw2;
        w3=w3+delw3;
        w4=w4+delw4;
        b=b+delb;
    end 
  wd=[delw1 delw2 delw3 delw4 delb];
  w=[w1 w2 w3 w4 b];  % to get final weights of this sample
  E(epoch)=current_error/samples;   % save valuse of each MSE E_bar=E/P 
  %% BEGIN MSE algorithm 
    if(epoch==1)     
        ebison=abs(current_error/samples);    % Get precision value =abs(Error(current)_Error(previous))
        temp=current_error/samples;
        current_error=0;      %reset accumlator 
    else
        ebison=abs(current_error/samples-temp);
        temp=current_error/samples;
        current_error=0;    %reset accumlator  
    end   
end
      plot(E); %Plot the graph of E vs Epoch
      grid on;
               %notify the user if network converges or not
   if(epoch~=epoch_limit)
       disp('precision goal met');   
   else
       disp('max epoch reached ');
   end

   %% Use final weights to test the network with data that the network havent seen before 
     
 T1=[0.9694 0.5427 0.6081 -0.1618 0.1870 0.4891 0.3777 1.1498 0.9325 0.5060 0.0497 0.4004 -0.1874 0.5060 1.6375 ];
 T2=[0.6909 1.3832 -0.9196 0.4694 -0.2578 -0.5276 2.0149 -0.4067 1.0950 1.3317 -2.0656 3.5369 1.3343 1.3317 -0.7911 ];
 T3=[0.4334 0.6390 0.5925 0.2030 0.6124 0.4378 0.7423 0.2469 1.0359 0.9222 0.6124 0.9766 0.5374 0.9222 0.7537 ];
 T4=[3.4965 4.0352 0.1016 3.0117 1.7749 0.6439 3.3932 1.5866 3.3591 3.7174 -0.6585 5.3532 3.2189 3.7174 0.5515];
 output_contener=zeros(1,15); %for memory allocation 
    for i=1:15
        
              output_contener(i)=w1*T1(i)+w2*T2(i)+w3*T3(i)+w4*T4(i)-b;     %If you want to check for numerical values        
  % output_contener=heaviside(w1*T1(i)+w2*T2(i)+w3*T3(i)+w4*T4(i)-b);    % classify directly
          
    end
   

