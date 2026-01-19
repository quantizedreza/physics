% Solving the Poisson equation in non-isotropic, non-homogenius materials
% using Finite ELement Method
% This example has been worked out for a homogenius material. 
% This code can be edited for non-homogenius materials (by using different material
% properties for differnet elements.

% (C) Reza Rahemi 2010

%% Initial Conditions and constants

epsilon_0=8.85*1e-12; %permitivity 
v0=1; %Volts
d=8*1e-2; %cm
rho=1e-8 ; %C.me-3

% Number of elements
N_e=7;
% Number of nodes
N_n=N_e+1;

%% element length

l=d/N_e;


%% k element matrix 
k_e=(epsilon_0/l)*[+1,-1;-1,+1];

%% f element matrix 
f_e=-l*rho/2*[1;1];

%Initialize the global matrix
k=zeros(N_n,N_n);
b=zeros(N_n,1);
elmconn=zeros(N_e,2); %Assembly matrix

for c=1:N_e;
elmconn(c,1)=c;
end

for c=1:N_e
elmconn(c,2)=c+1;
end


%Loop through the elements 

for e=1:N_e
    %Double loop through the local nodes of each element
    for i=1:2;
        for j=1:2;
            k(elmconn(e,i),elmconn(e,j))= k(elmconn(e,i),elmconn(e,j))+k_e(i,j);
        end
        b(elmconn(e,i))=b(elmconn(e,i))+f_e(i);
    end
end

%% solving for V (Potential)

% using the system of linear equations solver from matlab : Ax=B : x=B\A
%in this case, additional Drichlet boundary conditions that v(1,1)=0 and V=0 at the last node can be
%implemented so the first col and row of coef. matrix is eliminated. and
%the b matrix is updated according to b_i=b_i-ki1*V0 where V0=Vn and n is
%the row at which the Drichlet boundary condition is imposed upon. 

k_global=k(2:size(k,1)-1,2:size(k,2)-1);
b_global=b(2:size(b,1)-1,1);
for i=1:size(b_global,1);
b_global(i,1)=b(i+1,1)-k(i+1,1)*v0;
end

v=k_global\b_global;

v_matrix=[1;v;0]

plot(v_matrix)
