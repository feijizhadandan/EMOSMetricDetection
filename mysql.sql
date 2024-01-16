# 训练任务数据库
create table metric_train_task(
    id bigint auto_increment not null comment '数据库自增id',
    taskId varchar(100) not null comment '任务id(uuid)',
    createTime datetime not null comment '创建时间',
    startTime datetime not null comment '训练范围开始时间',
    endTime datetime not null comment '训练范围结束时间',
    onlineTask int not null comment '是否为在线实时监测',
    modelName varchar(100) not null comment '训练的模型名称',
    status varchar(20) not null comment '训练任务的运行状态(stop/running)',
    progress int not null comment '训练进度百分比',
    primary key (id)
);

# 模型数据库
create table model_metadata(
    id bigint auto_increment not null comment '数据库自增id',
    modelName varchar(100) not null comment '模型名称',
    createTime datetime not null comment '创建时间',
    meta json comment '训练参数(json)',
    primary key (id)
);

# 检测任务数据库
create table metric_detection_task(
    id bigint auto_increment not null comment '数据库自增id',
    taskId varchar(100) not null comment '任务id(uuid)',
    createTime datetime not null comment '检测任务创建时间',
    startTime datetime not null comment '检测开始时间',
    endTime datetime not null comment '检测结束时间',
    onlineTask int not null comment '是否为在线实时监测',
    modelName varchar(100) not null comment '用于检测的模型的名称',
    status varchar(20) not null comment '检测任务的运行状态(stop/running)',
    primary key (id)
);

# 检测任务数据库
create table metric_detection_result(
    id bigint auto_increment not null comment '数据库自增id',
    taskId varchar(100) not null comment '检测任务的id(uuid)',
    abnormal boolean not null comment '是否异常',
    maxScore double not null comment '任务检测范围内score最大值',
    threshold double not null comment '异常阈值score',
    primary key (id)
);
