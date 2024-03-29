% %Data Preprocessing
clear all
close all

load votes.dat;
load mpparty.dat;
load mpsex.dat;
load mpdistrict.dat

votes = reshape(votes,[349 31]);

%SOM
epochs = 200;
w = randi([2],100,31)/2;
% w = rand(100,31);
eta = 0.02;
sigma_start = 3;
sigma_end = 1;
tau = -epochs^2/log(sigma_end/sigma_start);
% w = rand(100,31)
%%
for i = 1:epochs
   for j = randperm(349)
        update_index = [];
        
%         [distance index] = min(sum((votes(j,:) - w).^2,2));

%         for k = 1:length(votes)
             for k = 1:length(w)
                 distance_input(k) = sum((votes(j,:) - w(k,:)).^2,2);
             end
            [~, index] = min(distance_input);
%         end
        
        
        distance = [];
%         index = 100;
        
        for l = 1:length(w)
            distance(l)  = max(abs((mod(abs(index-1),10)-mod(abs(l-1),10))),...
               abs(((index-1 - mod(index-1,10))/10)-((l-1 - mod(l-1,10))/10)));
        end
        sigma =sigma_start*exp(-i^2/tau);
        
%         distance_update = exp(-distance/(2*sigma^2)).*(distance <= sigma);
        distance_update = (distance <= sigma);
        
        w_delta = distance_update'.*(eta.*(votes(j,:) - w));
        w = w + (w_delta);
        winner(349*i+j)=index;
        
    
        
%         imagesc(reshape(distance_update,[10 10])') 
%         colorbar
%         drawnow
%         pause(1)
        
%         sigma
   end
   w_logg1(i) = w(1,1);
   w_logg2(i) = w(1,2);
   w_logg3(i) = w(50,1);
   w_logg4(i) = w(50,2);
%     imagesc(reshape((sum(w,2)),[10 10])')
%     imagesc(reshape(sum(votes*w',1),[10 10])')
%     imagesc(reshape(sum(w_delta,2),[10 10]),[-0.5 0.5])
%     imagesc(reshape(w(:,1),[10 10]),[-1 1])
%     colorbar
%     drawnow


%     sigma
%      imagesc(reshape(distance_update,[10 10])')
%         colorbar
%         drawnow
end
%%
% Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
% Use some color scheme for these different groups
for k = 1:length(votes)
    for i = 1:length(w)
        distance(i) = sum((votes(k,:) - w(i,:)).^2,2);        
    end
%     distance = votes(k,:)*w'
    [distance_ultimate(k) index_ultimate(k)] = min(distance);
end


%%
% [sorted sorted_index] = sort(distance_ultimate);

imagesc(reshape((sum(w,2)),[10 10])')
% [sorted index_ultimate] = sort(index_ultimate);

color_code = [0 0 0;0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250; ...
    0.4940 0.1840 0.5560; 0.4660 0.6740 0.1880; 0.3010 0.7450 0.9330; 0.6350 0.0780 0.1840];
title_names = ["No Party";"M"; "Fp";  "S"; "V"; "Mp"; "Kd"; "C"];

coord = zeros(100,2);
for i = 1:100
    num = i-1;
    coord(i,:) = [(num-mod(num,10))/10 mod(num,10)];
end


for i = 0:7
    figure()
    ind = index_ultimate(find(mpparty==i));
%     uni = find(mpparty(index_ultimate)==i);
    
    uni = unique(ind);
    cnt = histc(ind,uni);
    scatter(coord(uni,2),coord(uni,1),cnt*100,'markerfacecolor',[color_code(1+i,:)],...
        'markeredgecolor',[color_code(1+i,:)],'LineWidth',2)

%     scatter(coord(uni,2),coord(uni,1),'markerfacecolor',[color_code(1+i,:)],...
%         'markeredgecolor',[color_code(1+i,:)],'LineWidth',2)

%     imagesc(reshape(sum(votes(ind,:)*w',1),[10 10])')
    hold on
    axis([-1 10 -1 11])
    title(title_names(i+1,:))
end
%%
title_sex = ["Men","Women"]

for i = 0:1
    figure()
    ind = index_ultimate(find(mpsex==i));
%     uni = find(mpparty(index_ultimate)==i);
    
    uni = unique(ind);
    cnt = histc(ind,uni);
    scatter(coord(uni,2),coord(uni,1),cnt*100,'markerfacecolor',[color_code(1+i,:)],...
        'markeredgecolor',[color_code(1+i,:)],'LineWidth',2)

%     scatter(coord(uni,2),coord(uni,1),'markerfacecolor',[color_code(1+i,:)],...
%         'markeredgecolor',[color_code(1+i,:)],'LineWidth',2)

%     imagesc(reshape(sum(votes(ind,:)*w',1),[10 10])')
    hold on
    axis([-1 10 -1 11])
    title(title_sex(i+1))
end

% legend('No Party','M', 'Fp', 'S', 'V', 'Mp', 'Kd', 'C','location','northeastoutside')  
% axis([-0.1 1 -0.1 1])
%%
for i = 0:7
    ind = index_ultimate(find(mpparty==i));
    i
    ind
    mpparty(ind)
end










