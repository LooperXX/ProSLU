# ProSLU

`*.json` contains the utterance, intent, slots and supporting profile information (KG, UP and CA) for each sample.

## KG Information
- KG consists of multiple entities and use `；` to concatenate them
- each entity is represented by flattening its attributes
- KG information for each entity comes from the following open-source  knowledge graph
  - [CN-DBpedia](http://kw.fudan.edu.cn/apis/cndbpedia/)
  - [OwnThink](https://www.ownthink.com/docs/kg/)
  - ...

## UP & CA Information
```python
UP = {
    '音视频应用偏好': ['音乐类', '视频类', '有声读物类'],
    '出行交通工具偏好': ['地铁', '公交', '驾车'],
    '长途交通工具偏好': ['火车', '飞机', '汽车'],
    '是否有车': ['是', '否']
}

CA = {
    '移动状态': ['行走', '跑步', '静止', '汽车', '地铁', '高铁', '飞机', '未知'],
    '姿态识别': ['躺卧', '行走', '未知'],
    '地理围栏': ['家', '公司', '国内', '未知'],
    '户外围栏': ['户外', '室内', '未知']
}
```