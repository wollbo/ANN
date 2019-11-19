% %Data Preprocessing
clear all
close all

load votes.dat;
load mpparty.dat;
load mpsex.dat;
load mpdistrict.dat

% votes = reshape(votes,[349 31]);
votes = reshape(votes,[31 349])';

%SOM
epochs = 100;
w = rand(100,31);
eta = 0.2;
sigma_start = 4.5;
sigma_end = 1.5;
tau = -epochs^2/log(sigma_end/sigma_start);
count_winners = zeros(100,1);
%%
for i = 1:epochs
   sigma =(sigma_start*exp(-i^2/(2*tau^2)));
   for j = randperm(349)
%     for j = 1:349
        update_index = [];
             for k = 1:length(w)
                distance_input(k) = (-w(k,:)+votes(j,:))*(-w(k,:)+votes(j,:))';
             end
            [~, index] = min(distance_input);

        distance = zeros(length(w),2);

        for l = 1:length(w)
            distance(l,:)  = [abs((mod(abs(index-1),10)-mod(abs(l-1),10))),...
               abs(((index-1 - mod(index-1,10))/10)-((l-1 - mod(l-1,10))/10))];
        end
              
        distance_update = (distance(:,1)+distance(:,2))'<=round(sigma);%Manhattan
%         distance_update = (sqrt(distance(:,1).^2+distance(:,2).^2))'<=sigma;%Absolute Distance

        distance_update = exp(-sqrt(distance(:,1).^2+distance(:,2).^2)./(sigma^2)).*distance_update';
        
        update_index = find(distance_update);
        err = (votes(j,:) - w(update_index,:));
        
        w_delta = eta*err;
        w(update_index,:) = w(update_index,:) + (w_delta);
        winner(349*(i-1)+j)=index;
       
%Visualize the update region
%         imagesc(reshape(distance_update,[10 10])') 
%         colorbar
%         drawnow
%         pause(1)
        
        w_logg2(349*(i-1)+j) = (-w(55,:)+votes(55,:))*(-w(55,:)+votes(55,:))';
        w_logg4(349*(i-1)+j) =(-w(1,:)+votes(1,:))*(-w(1,:)+votes(1,:))';
   end
end

% u = unique(nonzeros(winner));
% c = histc((winner),u);
% count_winners(u) = c;
% imagesc(reshape(count_winners,[10 10]))
% colorbar
        
%%
% Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
% Use some color scheme for these different groups
for k = 1:length(votes)
    distance = zeros(1,length(w));
    for i = 1:length(w)
        distance(i) = (-w(i,:)+votes(k,:))*(-w(i,:)+votes(k,:))';        
    end
    [distance_ultimate(k) index_ultimate(k)] = min(distance);
end
%%
color_code = [0 0 0;0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250; ...
    0.4940 0.1840 0.5560; 0.4660 0.6740 0.1880; 0.3010 0.7450 0.9330; 0.6350 0.0780 0.1840];
title_names = ["No Party";"M"; "Fp";  "S"; "V"; "Mp"; "Kd"; "C"];

coord = zeros(100,2);
for i = 1:100
    num = i-1;
    coord(i,:) = [(num-mod(num,10))/10 mod(num,10)];
end


%%
subplot(2,4,1)

for i = 0:7
%     figure()
    subplot(2,4,i+1)
    temp = zeros(1,length(w));
    party_ind = find(mpparty==i);
    ind = index_ultimate(party_ind);
    
    uni = unique(ind);
    cnt = histc(ind,uni);
    
% scatter(coord(uni,2),coord(uni,1),cnt*100,'markerfacecolor',[color_code(1+i,:)],...
%         'markeredgecolor',[color_code(1+i,:)],'LineWidth',2)
    
    temp(uni)=cnt;
    imagesc(reshape(temp,[10 10]))
%     imagesc(sum(reshape(temp,[10 10]),2))
    hold on
%     axis([0 11 0 11])
    title(title_names(i+1,:))
%     axis equal
end
%%
title_sex = ["Men","Women"]

for i = 0:1
    subplot(1,2,i+1)
    temp = zeros(1,length(w));
%     figure()
    ind = index_ultimate(find(mpsex==i));
    
    uni = unique(ind);
    cnt = histc(ind,uni);
    
    temp(uni)=cnt;
    imagesc(reshape(temp,[10 10]))

    hold on
%     axis([-1 10 -1 11])
    title(title_sex(i+1))
end

% legend('No Party','M', 'Fp', 'S', 'V', 'Mp', 'Kd', 'C','location','northeastoutside')  
% axis([-0.1 1 -0.1 1])
%%
% title_sex = ["Men","Women"]
district = [1:29]

for i = 1:29
    subplot(5,6,i)
    temp = zeros(1,length(w));
%     figure()
    ind = index_ultimate(find(mpdistrict==i));
    
    uni = unique(ind);
    cnt = histc(ind,uni);
    
    temp(uni)=cnt;
    imagesc(reshape(temp,[10 10]))

    hold on
%     axis([-1 10 -1 11])
    text_title = [district(i)];
    title(text_title)
%     axis off
end
%%
% for i = 0:7
%     ind = index_ultimate(find(mpparty==i));
%     i
%     ind
%     mpparty(ind)
% end

%%
% figure()
% plot(w_logg4)
% hold on
% plot(w_logg2)
% 






