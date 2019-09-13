
%% plots inputs and firing rates over time for a single simulation
% define target colours and location
blue = [50 100 180]/255;
green = [120 180 120]/255;
target_r = pdist([0 0; targets(1,:)],'euclidean')/5; % target circle radius, plot as 1/10th of target distance

% plotting parameters
Imin = 0;
Imax = 110;

plot_space = [-.05 .04];% x/y space between subplots

% set up figure properties
set(0,'DefaultAxesBox', 'off',...,...
    'DefaultLineLineWidth',2,...
    'DefaultLineMarkerSize',8, ...
    'DefaultAxesLineWidth',2, ...
    'DefaultAxesFontSize',14,...
    'DefaultAxesLayer','top',...
    'DefaultAxesTickDir', 'in',...
    'DefaultAxesTickLength', [.025 1],...
    'defaultAxesTickDirMode', 'manual',...
    'DefaultAxesFontWeight','Bold',...
    'DefaultTextFontName', 'TimesNewRoman',...
    'DefaultFigureColor', 'w',...
    'DefaultAxesColor', 'w');


imageN=0;

for c = 1:length(trialIDs)
    
    tr = trialIDs(c);
    
    if CoMov(tr)==1
        CoMtype = 'CoMov';
    elseif CoMovInt(tr)==1
        CoMtype = 'CoMovInt';
    elseif CoInt(tr)==1
        CoMtype = 'CoInt';
    else
        CoMtype = 'no-CoM';
    end
    
    if error_final(tr,1) == 0 && (CoMt(tr) > 180 || isnan(CoMt(tr))) && ((CoInt(tr)==1 && error_final(tr,2) == 0) || (CoMovInt(tr)==1 && error_initial(tr,2) == 0 && error_final(tr,1) == 0) || (CoMov(tr)==1 && error_final(tr,1) == 0) || CoMn(tr) == 0)
        imageN=imageN+1;
        
        if ~exist(['resultFigures/cmaes_parameters2/' CoMtype num2str(tr) '.png'],'file')
                        
            
            %% plot network inputs:
            % sensory input and intentional strength (input to nodes 1-4: S1/S2 and I1/I2)
            figure('units','normalized','outerposition',[0 0 .75 1.2]);
            
            h1 = subplot(3,2,1); hold on
            fig_pos = get(h1,'Position');
            set(h1,'Position',[fig_pos(1) fig_pos(2)+plot_space(2) fig_pos(3:4)]);
            axis([0 t_fin(tr)*dt*1000 Imin Imax])
            xlabel('Time (ms)')
            ylabel('Input (Hz)')
            title('Input into I_{1}, I_{2} and S_{1}, S_{2}', 'FontSize', 16);
            plot((1:t_fin(tr))*dt*1000,Input(1,1:t_fin(tr),tr),'Color', blue);
            plot((1:t_fin(tr))*dt*1000,Input(2,1:t_fin(tr),tr),'Color', green);
            plot((1:t_fin(tr))*dt*1000,Input(3,1:t_fin(tr),tr),'Color', [.5 .5 .5]);
            plot((1:t_fin(tr))*dt*1000,Input(4,1:t_fin(tr),tr), 'Color', [0 0 0]);
            %hleg1 = legend('in_{I1}^{ }(blue)', 'in_{I2}^{ }(green)', 'in_{S1}^{ }(left)', 'in_{S2}^{ }(right)', 'Location', 'NorthEastOutside');
            %set(hleg1,'units', 'centimeters', 'position',[15 21 4.5 4], 'LineWidth', 1);
            set(gca, 'XTick', [0:200:t_fin(tr)], 'YTick', [0:20:100]);
            axis square;
            
            %% plot actual cost (distance to each target location)
            h2 = subplot(3,2,2); hold on
            fig_pos = get(h2,'Position');
            set(h2,'Position',[fig_pos(1)+plot_space(1) fig_pos(2)+plot_space(2) fig_pos(3:4)]);
            axis([ 0 t_fin(tr)*dt*1000 0 Imax])
            xlabel('Time (ms)')
            ylabel('Input (Hz)')
            title(['Input into C_{1} to C_{4}'], 'FontSize', 16);
            plot((1:t_fin(tr))*dt*1000,Input(9,1:t_fin(tr),tr),'Color', blue*1.4); hold on
            plot((1:t_fin(tr))*dt*1000,Input(10,1:t_fin(tr),tr),'Color', blue/1.4);
            plot((1:t_fin(tr))*dt*1000,Input(11,1:t_fin(tr),tr),'Color', green*1.4);
            plot((1:t_fin(tr))*dt*1000,Input(12,1:t_fin(tr),tr), 'Color', green/1.4);
            %hleg2 = legend('in_{C1}^{ }(blue-left)', 'in_{C2}^{ }(blue-right)', 'in_{C3}^{ }(green-left)', 'in_{C4}^{ }(green-right)', 'Location', 'NorthEastOutside');
            %set(hleg2,'units', 'centimeters', 'position',[30 21 4.5 4], 'LineWidth', 1);
            set(gca, 'XTick', [0:200:t_fin(tr)], 'YTick', [0:20:100]);
            axis square;
            
            
            %% plot firing rate of  nodes I1-I2 and S1-S2 (nodes 1:4)
            % figure('units','normalized','outerposition',[.25 0 .3 1]);
            h3 = subplot(3,2,3); hold on
            fig_pos = get(h3,'Position');
            set(h3,'Position',[fig_pos(1) fig_pos(2) fig_pos(3:4)]);
            axis([0 t_fin(tr)*dt*1000 Imin Imax])
            xlabel('Time (ms)')
            ylabel('Firing rate (Hz)')
            title('Firing rates of I_{1}, I_{2} and S_{1}, S_{2}', 'FontSize', 16);
            plot((1:t_fin(tr))*dt*1000,r(1,1:t_fin(tr),tr),'Color', blue);
            plot((1:t_fin(tr))*dt*1000,r(2,1:t_fin(tr),tr),'Color', green);
            plot((1:t_fin(tr))*dt*1000,r(3,1:t_fin(tr),tr),'Color', [.5 .5 .5]);
            plot((1:t_fin(tr))*dt*1000,r(4,1:t_fin(tr),tr),'Color', [0 0 0]);
            %hleg3 = legend('I_{1}^{ }(blue)', 'I_{2}^{ }(green)', 'S_{1}^{ }(left)', 'S_{2}^{ }(right)', 'Location', 'NorthEastOutside');
            %set(hleg3,'units', 'centimeters', 'position',[15 12 4.5 4], 'LineWidth', 1);
            set(gca, 'XTick', [0:200:t_fin(tr)], 'YTick', [0:20:100]);
            axis square;
            
            
            %% plot firing rate of cost nodes C1-C4 (nodes 9:12)
            % figure('units','normalized','outerposition',[.25 0 .3 1]);
            h4 = subplot(3,2,4); hold on
            fig_pos = get(h4,'Position');
            set(h4,'Position',[fig_pos(1)+plot_space(1) fig_pos(2) fig_pos(3:4)]);
            axis([0 t_fin(tr)*dt*1000 Imin Imax])
            xlabel('Time (ms)')
            ylabel('Firing rate (Hz)')
            title(['Firing rates of C_{1} to C_{4}'], 'FontSize', 16);
            plot((1:t_fin(tr))*dt*1000,r(9,1:t_fin(tr),tr),'Color', blue*1.4);
            plot((1:t_fin(tr))*dt*1000,r(10,1:t_fin(tr),tr),'Color', blue/1.4);
            plot((1:t_fin(tr))*dt*1000,r(11,1:t_fin(tr),tr),'Color', green*1.4);
            plot((1:t_fin(tr))*dt*1000,r(12,1:t_fin(tr),tr),'Color', green/1.4);
            %hleg4 = legend('C_{1}^{ }(blue-left)', 'C_{2}^{ }(blue-right)', 'C_{3}^{ }(green-left)', 'C_{4}^{ }(green-right)', 'Location', 'NorthEastOutside');
            %set(hleg4,'units', 'centimeters', 'position',[30 12 4.5 4], 'LineWidth', 1);
            set(gca, 'XTick', [0:200:t_fin(tr)], 'YTick', [0:20:100]);
            axis square;
            
            
            %% plot firing rate of action nodes A1-A4 (nodes 5:8)
            % figure('units','normalized','outerposition',[.25 0 .3 1]);
            h5 = subplot(3,2,5); hold on
            fig_pos = get(h5,'Position');
            set(h5,'Position',[fig_pos(1) fig_pos(2)-plot_space(2) fig_pos(3:4)]);
            axis([0 t_fin(tr)*dt*1000 Imin Imax])
            xlabel('Time (ms)')
            ylabel('Firing rate (Hz)')
            title(['Firing rates of A_{1} to A_{4}'], 'FontSize', 16);
            plot((1:t_fin(tr))*dt*1000,repmat(thresh,1,length(1:t_fin(tr))),'k-.'); % threhsold for action execution
            plot((1:t_fin(tr))*dt*1000,r(5,1:t_fin(tr),tr),'Color', blue*1.4);
            plot((1:t_fin(tr))*dt*1000,r(6,1:t_fin(tr),tr),'Color', blue/1.4);
            plot((1:t_fin(tr))*dt*1000,r(7,1:t_fin(tr),tr),'Color', green*1.4);
            plot((1:t_fin(tr))*dt*1000,r(8,1:t_fin(tr),tr),'Color', green/1.4);
            %hleg5 = legend('Threshold', 'A_{1}^{ }(blue-left)','A_{2}^{ }(blue-right)', 'A_{3}^{ }(green-left)', 'A_{4}^{ }(green-right)', 'Location', 'NorthEastOutside');
            %set(hleg5,'units', 'centimeters', 'position',[15 3 4.5 5], 'LineWidth', 1);
            set(gca, 'XTick', [0:200:t_fin(tr)], 'YTick', [0:20:100]);
            axis square;
            
            
            %% plot actual movement trajectory and target circles
            h6 = subplot(3,2,6); hold on;
            fig_pos = get(h6,'Position');
            set(h6,'Position',[fig_pos(1)+plot_space(1) fig_pos(2)-plot_space(2) fig_pos(3:4)]);
            axis([-(abs(targets(1,1))+target_r*2) abs(targets(1,1))+target_r*2 -(abs(targets(1,2))+target_r*2) abs(targets(1,2))+target_r*2])
            title(['Movement trajectory'], 'FontSize', 16);
            xlabel('x'); ylabel('y')
            plot(0,0,'k+');
            ht(1) = plot(targets(1,1), targets(1,2), 'o', 'MarkerSize', 15, 'MarkerFaceColor', blue, 'MarkerEdgeColor', [0 0 0], 'LineWidth', 1);
            ht(2) = plot(targets(2,1), targets(2,2), 'o', 'MarkerSize', 15, 'MarkerFaceColor', blue, 'MarkerEdgeColor', [0 0 0], 'LineWidth', 1);
            ht(3) = plot(targets(3,1), targets(3,2), 'o', 'MarkerSize', 15, 'MarkerFaceColor', green, 'MarkerEdgeColor', [0 0 0], 'LineWidth', 1);
            ht(4) = plot(targets(4,1), targets(4,2), 'o', 'MarkerSize', 15, 'MarkerFaceColor', green, 'MarkerEdgeColor', [0 0 0], 'LineWidth', 1);
            
            % find trajectories during non-decision time
            nd_time = find(pos(1,:,tr),sum(nd_t)); % movement driven during nd time
            
            if CoMov(tr)==1 % if CoMov: for visualisation, replot initial trajectories slightly shifted so that initial/final trajectory don't overlap perfectly
                initialTraj = find(pos(1,:,tr),round(CoMt(tr)));
                plot(pos(1,initialTraj,tr), 1.1*pos(2,initialTraj,tr)-5, 'k-'); % plot initial trajectory (before CoM)
                plot(pos(1,setdiff(nd_time,initialTraj),tr), pos(2,setdiff(nd_time,initialTraj),tr)+10, 'k-'); % plot trajectory after CoM (but before end of nd time)
                plot(pos(1,nd_time(end)+1:20:end,tr), pos(2,nd_time(end)+1:20:end,tr)+10, 'k.', 'MarkerSize',1); % plot final trajectory after nd time
            else
                plot(pos(1,nd_time,tr), pos(2,nd_time,tr), 'k-');
                plot(pos(1,nd_time(end)+1:20:end,tr), pos(2,nd_time(end)+1:20:end,tr), 'k.', 'MarkerSize',1);
            end
            
            axis square;
            
            %%% save figure with export_fig (https://www.mathworks.com/matlabcentral/fileexchange/23629-export_fig)
            %export_fig(['output/' CoMtype num2str(tr)],'-png','-r1000'); 
            
            pause
            clf
            %close all
        end
    end
end