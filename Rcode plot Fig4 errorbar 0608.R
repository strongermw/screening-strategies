#format <- ".pdf"
setwd("D:/FDU program/COVID-19/研究 筛查检测策略/可视化/v10 可视化")
library(patchwork)
library(ggpubr)
data <- read.csv(paste0("compartments_time series_sim1000_0606.csv"))
data1 <- read.csv(paste0("cumulative & maximum data_sim1000_0606.csv"))

#dt <- read.csv("data_eachsim_0530.csv")

stra_list <- read.csv("Strategy&Scenario_0608.csv")
data <- merge(data,stra_list,by=c("stra_file","scen_file"),all.x=T)
data1 <- merge(data1,stra_list,by=c("stra_file","scen_file"),all.x=T)

#dt$strategy <- paste0(dt$stra_name," - ",dt$scen_name)
data$strategy <- paste0(data$stra_name," - ",data$scen_name)
data1$strategy <- paste0(data1$stra_name," - ",data1$scen_name)

dt_l <- data %>% filter(scen_file=="ScenarioA5_day1~4" 
                        | scen_file=="ScenarioA18_day1~7"
                        | scen_file == "ScenarioB1_day1~7"
                        | scen_file=="ScenarioC1_day1~7")


p_l <- ggplot(dt_l,aes(x=time))+
  #geom_ribbon(aes(ymin=infected_25,ymax=infected_75,group=strategy,fill=strategy),alpha = 0.06)+
  #scale_fill_manual(values=colors2)+
  geom_line(aes(y=infected_m,group=strategy,color=strategy),size=1)+
  scale_color_nejm()+
  #scale_color_manual(values=colors2)+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        legend.text = element_text(size=12))+
  guides(color=guide_legend(nrow=1))+
  labs(x = "time", y = "cases",tag="A") ;p_l

dt_h <- data1 %>% filter(scen_file=="ScenarioA5_day1~4" 
                         | scen_file=="ScenarioA18_day1~7"
                         | scen_file == "ScenarioB1_day1~7"
                         | scen_file=="ScenarioC1_day1~7")

dt_h$strategy <- paste0(dt_h$stra_name," - ",dt_h$scen_name)
colnames(dt_h)
p_ir <- ggplot(dt_h)+
  #geom_boxplot(aes(x=factor(strategy),y=proportion_infected_sum,color=strategy))+
  geom_point(aes(x=factor(strategy),y=proportion_infected_sum_m,color=strategy),shape = 18,size=5)+
  geom_errorbar(aes(x=factor(strategy),ymin=proportion_infected_sum_25,ymax=proportion_infected_sum_75,color=strategy),width=0.3,size=1.5)+
  scale_color_nejm()+
  theme_classic()+
  theme(legend.position = "none",
        legend.title = element_blank(),
        axis.text.x = element_blank())+
  guides(color=guide_legend(nrow=1))+
  labs(x = "strategy", y = "cumulative infection rate",tag="B");p_ir

p_test <- ggplot(dt_h)+
  #geom_boxplot(aes(x=factor(strategy),y=tested_sum,color=strategy))+
  geom_point(aes(x=factor(strategy),y=tested_sum_m,color=strategy),shape = 18,size=5)+
  geom_errorbar(aes(x=factor(strategy),ymin=tested_sum_25,ymax=tested_sum_75,color=strategy),width=0.3,size=1.5)+
  scale_color_nejm()+
  theme_classic()+
  theme(legend.position = "none",
        legend.title = element_blank(),
        axis.text.x = element_blank())+
  labs(x = "strategy", y = "number of tests",tag="C");p_test


#(p_l | (p_ir / p_test)) 
p_ir14 <- ggarrange(p_l,p_ir,p_test,nrow=1,common.legend=TRUE,legend="top");p_ir14
#ggsave(paste0("./Plots/","Fig 14pct boxplot"," 0530.png"),width = 13,height = 5,dpi = 500)



#####################################################################

dt_l <- data %>% filter(scen_file=="ScenarioA6_day1~3" 
                        | scen_file=="ScenarioB2_day1~6"
                        | scen_file == "ScenarioC2_day1~6"
                        | scen_file=="ScenarioA19_day1~6"
                        | scen_file=="ScenarioB18_day1~7"
                        | scen_file=="ScenarioC18_day1~7")

p_l <- ggplot(dt_l,aes(x=time))+
  #geom_ribbon(aes(ymin=infected_25,ymax=infected_75,group=strategy,fill=strategy),alpha = 0.06)+
  #scale_fill_manual(values=colors2)+
  geom_line(aes(y=infected_m,group=strategy,color=strategy),size=1)+
  #scale_color_manual(values=colors2)+
  scale_color_lancet()+
  theme_classic()+
  theme(legend.position = "top",                     
        legend.background = element_blank(), 
        legend.key = element_blank(),
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        legend.text = element_text(size=12))+
  guides(color=guide_legend(nrow=2))+
  labs(x = "time", y = "cases",tag="D") ;p_l


dt_h <- data1 %>% filter(scen_file=="ScenarioA6_day1~3" 
                      | scen_file=="ScenarioB2_day1~6"
                      | scen_file == "ScenarioC2_day1~6"
                      | scen_file=="ScenarioA19_day1~6"
                      | scen_file=="ScenarioB18_day1~7"
                      | scen_file=="ScenarioC18_day1~7")

dt_h$strategy <- paste0(dt_h$stra_name," - ",dt_h$scen_name)
#colnames(dt_h)

p_ir <- ggplot(dt_h)+
  #geom_boxplot(aes(x=factor(strategy),y=proportion_infected_sum,color=strategy))+
  geom_point(aes(x=factor(strategy),y=proportion_infected_sum_m,color=strategy),shape = 18,size=5)+
  geom_errorbar(aes(x=factor(strategy),ymin=proportion_infected_sum_25,ymax=proportion_infected_sum_75,color=strategy),width=0.3,size=1.5)+
  scale_color_lancet()+
  theme_classic()+
  theme(legend.position = "none",
        legend.title = element_blank(),
        axis.text.x = element_blank())+
  guides(color=guide_legend(nrow=1))+
  labs(x = "strategy", y = "cumulative infection rate",tag="E");p_ir

p_test <- ggplot(dt_h)+
  #geom_boxplot(aes(x=factor(strategy),y=tested_sum,color=strategy))+
  geom_point(aes(x=factor(strategy),y=tested_sum_m,color=strategy),shape = 18,size=5)+
  geom_errorbar(aes(x=factor(strategy),ymin=tested_sum_25,ymax=tested_sum_75,color=strategy),width=0.3,size=1.5)+
  scale_color_lancet()+
  theme_classic()+
  theme(legend.position = "none",
        legend.title = element_blank(),
        axis.text.x = element_blank())+
  labs(x = "strategy", y = "number of tests",tag="F");p_test



#(p_l | (p_ir / p_test)) 
p_ir34 <- ggarrange(p_l,p_ir,p_test,nrow=1,common.legend=TRUE,legend="top",legend.grob=get_legend(p_test));p_ir34

ggarrange(p_ir14,p_ir34,nrow=2)

ggsave(paste0("./Plots/","Fig4 14&34pct boxplot","_",date,format),width = 15,height = 10,dpi = 500)
