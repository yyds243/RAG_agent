from neo4j import GraphDatabase
from knowledge.utils.neo4j_utils import get_neo4j_driver

# ========== 1. 建立连接 ==========
neo4j_driver = get_neo4j_driver()

driver = GraphDatabase.driver(URI, auth=AUTH)
driver.verify_connectivity()
print("连接成功！")

#
# # ========== 2. 写入 ==========
# with driver.session(database=DATABASE) as session:
#     # 先清空
#     session.run("MATCH (n) DETACH DELETE n").consume()
#
#     # 创建节点
#     session.run(
#         "CREATE (:Customer {name: $name, age: $age, vip: $vip})",
#         name="张三", age=28, vip=True,
#     ).consume()
#
#     session.run(
#         "CREATE (:Customer {name: $name, age: $age, vip: $vip})",
#         name="李四", age=35, vip=True,
#     ).consume()
#
#     # 创建关系
#     session.run("""
#         MATCH (a:Customer {name: $a}), (b:Customer {name: $b})
#         CREATE (a)-[:FRIEND]->(b)
#     """, a="张三", b="李四").consume()
#
#     print("写入完成！")

# # ========== 3. 查询 ==========
# with driver.session(database=DATABASE) as session:
#     result = session.run(
#         "MATCH (c:Customer) RETURN c.name AS name, c.age AS age, c.vip AS vip"
#     )
#     for record in result:
#         print(record.data())
#     # 输出: {'name': '张三', 'age': 28, 'vip': True}
#     #       {'name': '李四', 'age': 35, 'vip': True}


# ========== 4. 更新 ==========
# with driver.session(database=DATABASE) as session:
#     result = session.run(
#         "MATCH (c:Customer {name: $name}) SET c.city = $city RETURN c",
#         name="张三", city="北京",
#     )
#     print(result.single()["c"])  # {'c.name': '张三', 'c.city': '北京'}



# 2. 定义 Cypher 语句
clear_cypher = "MATCH (n) DETACH DELETE n"
setup_cypher = """
    MERGE (c1:Chunk {id: 'c1', text: 'Chunk 1: RS-12数字万用表的电池后盖由两颗十字螺丝固定。'})
    MERGE (c2:Chunk {id: 'c2', text: 'Chunk 2: 拆卸本设备的十字螺丝时，必须使用带有绝缘手柄的金属十字螺丝刀。'})
    MERGE (c3:Chunk {id: 'c3', text: 'Chunk 3: 警告：使用任何金属工具接触仪表内部前，必须先断开测试表笔，否则有触电危险。'})

    MERGE (cover:Entity {name: '电池后盖'})
    MERGE (screw:Entity {name: '十字螺丝'})
    MERGE (tool:Entity {name: '金属螺丝刀'})
    MERGE (warning:Entity {name: '警告-断开测试表笔'})

    MERGE (cover)-[:SECURED_BY]->(screw)
    MERGE (screw)-[:REQUIRES_TOOL]->(tool)
    MERGE (tool)-[:HAS_WARNING]->(warning)
"""

def run_setup():
    with driver.session() as session:
        # 第一步：先单独清理
        session.run(clear_cypher)
        # 第二步：再单独写入
        res = session.run(setup_cypher)
        res.consume()

        print("✅ 执行成功！节点和关系已创建。")
    driver.close()

if __name__ == '__main__':
    run_setup()