import heapq
"""
遍历版本：遍历每一个墙壁进行爆破，之后推箱（保底）
"""

class GameSymbol:
    def __init__(self):
        self.wall = "#"
        self.box = "$"
        self.car = "@"
        self.goal = "."
        self.bomb = "*"
        self.ground = " "       # 待更改


game_symbol = GameSymbol()

WALL = game_symbol.wall
BOX = game_symbol.box
TARGET = game_symbol.goal
PLAYER = game_symbol.car
BOMB = game_symbol.bomb

class AStarSolver:
    """
    全排列模拟求解器 (Strict Target Mode)
    修复问题：防止箱子在推向目标A的过程中，意外掉进目标B。
    解决方法：在单次寻路时，将“非当前目标的所有其他剩余目标”也视为墙壁。
    新增功能：推箱子遇单块墙时，忽略该墙并标记爆破点，仅允许一次忽略
    """

    def __init__(self, raw_map, start_point, difficulty):
        self.raw_map = raw_map
        self.height = len(raw_map)
        self.width = len(raw_map[0])
        self.walls = set()

        self.difficulty = difficulty

        self.initial_boxes = []
        self.initial_targets = []
        self.start_player = start_point
        self.initial_bombs = []

        # # 获取对应关系
        # if self.difficulty != 1:
        #     self.box_order = box_order
        #     self.target_order = target_order

        for y, row in enumerate(raw_map):
            for x, char in enumerate(row):
                if char == WALL:
                    self.walls.add((x, y))
                elif char == BOX:
                    self.initial_boxes.append((x, y))
                elif char == TARGET:
                    self.initial_targets.append((x, y))
                # elif char == PLAYER:
                #     self.start_player = (x, y)
                elif char == BOMB:
                    self.initial_bombs.append((x, y))

        self.current_invest_pos = self.start_player  #侦察阶段使用

    def solve(self, box_order, target_order):  # 返回：坐标列表
        n_boxes = len(self.initial_boxes)
        n_targets = len(self.initial_targets)

        if n_boxes != n_targets:
            print("错误：箱子与目标数量不符")
            return []

        task_sequence = []

        if self.difficulty == 1:
            box_permutations = list(self.permutations(range(n_boxes)))
            target_permutations = list(self.permutations(range(n_targets)))
            best_total_steps = float('inf')
            best_full_path = []
            for box_order in box_permutations:
                for target_order in target_permutations:
                    # 构建任务序列
                    task_sequence = []
                    for i in range(n_boxes):
                        b_idx = box_order[i]
                        t_idx = target_order[i]
                        task_sequence.append((self.initial_boxes[b_idx], self.initial_targets[t_idx]))
                    path, steps, success = self._simulate_sequence(task_sequence)
                    if success:
                        if steps < best_total_steps:
                            best_total_steps = steps
                            best_full_path = path
            if best_full_path:
                print(f"最优解找到! 总步数: {best_total_steps}")
                return best_full_path  # 返回有序坐标列表
            else:
                print("所有路径均不可行 (可能存在死锁或无法避开的陷阱)")
                return []

        else:
            for i in range(n_boxes):
                b_idx = box_order[i]
                t_idx = target_order[i]
                task_sequence.append((self.initial_boxes[b_idx], self.initial_targets[t_idx]))

            path, steps, success = self._simulate_sequence(task_sequence) #总步数，自然包括推炸弹用的步数
            if success:
                print(f"获取路径成功，总步数：{steps}")
                return path
            else:
                print("所有路径均不可行 (可能存在死锁或无法避开的陷阱)")
                return []


    def _simulate_sequence(self, sequence):
        """模拟执行序列（处理炸弹+爆破逻辑）"""
        # 复制炸弹列表，避免修改原初始数据
        bombs = set(self.initial_bombs)
        # 序列障碍物(本次序列全局共享)

        # 先筛选有效墙块（零成本过滤，减少遍历量）
        all_walls = set(self.walls)  # 过滤边缘
        valid_walls = [
            (x,y) for (x,y) in all_walls
            if 0<x<self.width-1 and 0<y<self.height-1
        ]

        start_player = self.current_invest_pos

        ####寻路逻辑###
        # 1.无炸弹，常规推箱子
        if not bombs:

            sequence_obstacles = set(self.walls)
            active_boxes = set(b for b, t in sequence)
            active_targets = set(self.initial_targets)
            current_player = start_player
            full_path = {}
            dict_step = 0
            total_steps = 0
            sim_px = 0
            sim_py = 0

            print("场上无炸弹，正常推箱")
            for target_box, target_pos in sequence:#对每一个（箱子，目标点）对
                #1. 其他箱子是墙
                for b in active_boxes:
                    if b != target_box:
                        sequence_obstacles.add(b)
                # 2. 其他目标也是墙 (防止误触)
                if self.difficulty == 1:
                    for t in active_targets:
                        if t != target_pos:
                            sequence_obstacles.add(t)

                path = self._solve_single_box(current_player, target_box, target_pos, sequence_obstacles)
                if path is None:
                    print("没有找到正常路径！")
                    return [], 0, False
                # 更新状态
                Path = []
                Path.append(current_player)
                Path.extend(self.actions_to_paths(current_player,path))
                full_path[dict_step] = Path
                dict_step += 1
                total_steps += len(path)

                # 模拟玩家移动
                sim_px, sim_py = current_player     #玩家执行前前位置
                for dx, dy in path:
                    sim_px += dx
                    sim_py += dy
                current_player = (sim_px, sim_py)   #玩家执行后位置
                # 移除已完成的箱子和目标
                active_boxes.remove(target_box)
                active_targets.remove(target_pos)
                # 从障碍中去除
                #1. 其他箱子
                for b in active_boxes:
                    if b != target_box:
                        sequence_obstacles.remove(b)
                #2. 其他目标
                for t in active_targets:
                    if t != target_pos:
                        sequence_obstacles.discard(t)
            # 最后回到发车点
            after_path = self._astar_move_to_pos(current_player,self.start_player,sequence_obstacles)
            After_path = []
            After_path.append(current_player)
            After_path.extend(self.actions_to_paths(current_player,after_path))
            full_path[dict_step] = After_path

            for i in range(dict_step):
                total_steps += len(full_path[i])
            return full_path, total_steps, True

        # 2.场上有炸弹
        else:
            # 选墙爆破
            print("场上有炸弹，进入爆破模式")
            best_full_path = []                     #初始化最优解
            best_total_steps = float('inf')         #初始化最优步数
            # wall_combinations = itertools.combinations(valid_walls, n_bombs)    #从可炸墙中选择n(炸弹数量)个墙块
            def _get_bomb_nearby_walls(bomb_pos, valid_walls, radius=2):
                bx, by = bomb_pos
                nearby_walls = []
                # 遍历5×5范围（bx-2 ~ bx+2, by-2 ~ by+2）
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        check_x = bx + dx
                        check_y = by + dy
                        # 检查该坐标是否是有效墙块
                        if (check_x, check_y) in valid_walls:
                            nearby_walls.append((check_x, check_y))
                return nearby_walls

            # 为每个炸弹筛选5×5范围内的有效墙块
            bomb_nearby_walls = {}
            all_nearby_walls = set()
            for bomb in self.initial_bombs:
                nearby_walls = _get_bomb_nearby_walls(bomb, valid_walls)
                bomb_nearby_walls[bomb] = nearby_walls
                all_nearby_walls.update(nearby_walls)
            all_nearby_walls = list(all_nearby_walls)

            bomb_list = list(self.initial_bombs)
            valid_combinations = self._generate_all_valid_combos(bomb_list,bomb_nearby_walls)


            # 新增：为每个基础combo生成推动顺序的全排列，扩展到最终列表
            final_valid_combinations = []
            for base_combo in valid_combinations:
                # 对当前基础combo生成所有推动顺序的全排列
                # 比如base_combo是[(B1,W1),(B2,W2),(B3,W3)]，会生成6种不同顺序的排列
                permutations_of_combo = list(self.permutations(base_combo))
                # 将所有排列加入最终列表
                final_valid_combinations.extend(permutations_of_combo)

            # 替换原有遍历对象：现在final_valid_combinations包含所有combo+所有推动顺序
            valid_combinations = final_valid_combinations

            # 3. 遍历组合（替代原selected_walls/wall_perms/bomb_perms嵌套）
            for combo in valid_combinations:
                # 构建任务序列(一次完整的任务)(相当于该函数的无炸弹部分)
                # 单次任务初始化
                active_targets = set(self.initial_targets)
                active_boxes = set(b for b, t in sequence)
                active_bombs = set(self.initial_bombs)
                bomb_obstacles = set(self.walls)        #初始化爆破流程专属障碍
                current_player = start_player
                full_path = []
                total_steps = 0
                sim_px = 0
                sim_py = 0
                # 箱子和目标在所有爆破完成之前都暂时视为墙壁
                for b in active_boxes:
                    bomb_obstacles.add(b)
                for t in active_targets:
                    bomb_obstacles.add(t)
                path1 = []
                path2 = []

                for target_bomb, target_wall in combo:      # 一次任务中，对每个(炸弹，爆破点)对进行移动
                    # 模拟  #TODO
                    print(f"当前炸弹：{target_bomb} 当前爆破点：{target_wall}")
                    # 因为是一次模拟，所以变量应当在这里创建
                    #1. 其他炸弹是墙
                    for bo in active_bombs:
                        if bo != target_bomb:
                            bomb_obstacles.add(bo)
                    # # 2. 其他墙块也是墙 (防止误触)
                    # for bo,w in combo:
                    #     if w != target_wall:
                    #         bomb_obstacles.add(w)
                    path = self._solve_single_box(current_player, target_bomb, target_wall, bomb_obstacles)
                    if path is None: #如果推不到，更换序列
                        print(f"无法完成推炸弹{target_bomb}")
                        continue
                    # 更新状态
                    path1.extend(path)
                    # 模拟玩家移动
                    sim_px, sim_py = current_player     #玩家执行前前位置
                    for dx, dy in path:
                        sim_px += dx
                        sim_py += dy
                    current_player = (sim_px, sim_py)   #玩家执行后位置
                    # 移除已爆破的炸弹，墙块
                    active_bombs.remove(target_bomb)
                    for i in (-1,0,1):
                        for j in (-1,0,1):
                            broken_wall = (target_wall[0]+i,target_wall[1]+j)
                            if broken_wall in valid_walls:# 外墙不可爆破
                                bomb_obstacles.discard(broken_wall)
                    # 从障碍中去除其余炸弹
                    for bo in active_bombs:
                        if bo != target_bomb:
                            bomb_obstacles.remove(bo)
                    print(f"推炸弹{target_bomb}成功")
                    # # 2.其他元素
                    # for b in active_boxes:
                    #     bomb_obstacles.remove(b)
                    # for t in active_targets:
                    #     bomb_obstacles.remove(t)
                # 如果推所有炸弹失败(对应上述path None)
                if not path1:
                    continue
                # 如果成功推到所有炸弹,进行常规推箱(复制第一种情况)
                else:
                    # 地图更新,将箱子、目标从障碍物中取消
                    print("准备进行常规推箱")
                    for b in active_boxes:
                        bomb_obstacles.remove(b)
                    for t in active_targets:
                        bomb_obstacles.remove(t)
                    for target_box, target_pos in sequence:#对每一个（箱子，目标点）对进行推箱
                        print(f"当前序列：箱子:{target_box},目标点:{target_pos}")

                        #1. 其他箱子是墙
                        for b in active_boxes:
                            if b != target_box:
                                bomb_obstacles.add(b)
                        # 2. 其他目标也是墙 (防止误触)
                        for t in active_targets:
                            if t != target_pos:
                                bomb_obstacles.add(t)
                        # 3. 剩下的炸弹也是墙
                        for bo in active_bombs:
                            bomb_obstacles.add(bo)

                        path = self._solve_single_box(current_player, target_box, target_pos, bomb_obstacles)
                        if path is None:
                            print("没有找到正常路径！/n")
                            break
                        # 更新状态
                        path2.extend(path)
                        # 模拟玩家移动
                        sim_px, sim_py = current_player     #玩家执行前前位置
                        for dx, dy in path:
                            sim_px += dx
                            sim_py += dy
                        current_player = (sim_px, sim_py)   #玩家执行后位置
                        # 移除已完成的箱子和目标
                        active_boxes.remove(target_box)
                        active_targets.remove(target_pos)
                        # 从障碍中去除
                        #1. 其他箱子
                        for b in active_boxes:
                            if b != target_box:
                                bomb_obstacles.remove(b)
                        # 2. 其他目标
                        for t in active_targets:
                            if t != target_pos:
                                bomb_obstacles.remove(t)

                    if not path2:
                        continue
                    else:
                        full_path = path1 + path2
                        steps = len(full_path)
                        print("发现一条成功路径！")
                        if steps < best_total_steps:
                            best_total_steps = steps
                            best_full_path = full_path
                            print(f"发现更优解: 步数 {steps}/n")


            if best_full_path:
                real_path = best_full_path
                total_steps = len(real_path)
                print(f"最优解找到! 总步数: {total_steps}")
                return real_path,total_steps,True
            else:
                print("所有路径均不可行，任务失败")
                return [],0,False



    def _solve_single_box(self, start_player, target_box, target_pos, obstacles):
        """单箱子 A* 寻路（基础推箱逻辑，无爆破）"""
        pq = []
        # 启发函数：箱子到目标点的曼哈顿距离
        h = abs(target_box[0] - target_pos[0]) + abs(target_box[1] - target_pos[1])
        heapq.heappush(pq, (h, 0, start_player, target_box, []))

        visited = set()
        visited.add((start_player, target_box))

        max_nodes = 500
        nodes = 0
        dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 下,上，左，右

        while pq:
            f, g, (px, py), (bx, by), path = heapq.heappop(pq)
            nodes += 1
            if nodes > max_nodes:
                print("单箱寻路节点超限，返回None")
                return None
            # 箱子到达目标，返回路径
            if (bx, by) == target_pos:
                return path
            # 遍历四个移动方向
            for dx, dy in dirs:
                nx, ny = px + dx, py + dy
                # 玩家撞墙检查（不可穿过障碍物）
                if (nx, ny) in obstacles :
                    continue
                new_box_pos = (bx, by)
                # 推箱子逻辑
                if (nx, ny) == (bx, by):
                    nbx, nby = bx + dx, by + dy
                    # 箱子新位置不可是障碍物
                    if (nbx, nby) in obstacles:
                        if (nbx , nby) != target_pos:
                            continue
                    new_box_pos = (nbx, nby)
                # 状态去重（玩家位置+箱子位置）
                new_player = (nx, ny)
                state = (new_player, new_box_pos)
                if state in visited:
                    continue
                visited.add(state)
                # 更新代价，加入优先队列
                new_g = g + 1
                new_h = abs(new_box_pos[0] - target_pos[0]) + abs(new_box_pos[1] - target_pos[1])   #TODO#可以加入玩家与箱子的距离
                heapq.heappush(pq, (new_g + new_h * 1.2, new_g, new_player, new_box_pos, path + [(dx, dy)]))
        return None

    # 核心模块：生成所有合法的顺序组合（逐个校验3×3范围）
    def _generate_all_valid_combos(self, bomb_list, bomb_nearby_walls):
        """
        生成所有合法组合，逻辑严格匹配你的需求：
        1. 按炸弹顺序（炸弹1→炸弹2→...→炸弹N）遍历
        2. 炸弹1：遍历所有5×5可炸块，逐个选取
        3. 炸弹2：遍历所有5×5可炸块，仅选不在炸弹1所选墙块3×3范围内的，逐个选取
        4. 后续炸弹：同理，仅选不在前面所有炸弹所选墙块3×3范围内的
        5. 最终返回所有合法的顺序组合列表
        :param bomb_list: 按顺序排列的炸弹列表 [(x1,y1), (x2,y2), ...]
        :param bomb_nearby_walls: 每个炸弹的5×5可炸块字典 {炸弹位置: [墙块列表], ...}
        :return: 所有合法组合列表 [[(炸弹1,墙块1), (炸弹2,墙块2), ...], ...]
        """
        all_valid_combos = []  # 存储所有合法的combo

        # 递归遍历所有可能的顺序组合，实时校验3×3范围
        def backtrack(bomb_index, selected_pairs, occupied_3x3_areas):
            """
            :param bomb_index: 当前处理的炸弹索引（0=第一个，1=第二个...）
            :param selected_pairs: 已选的(炸弹,墙块)配对列表
            :param occupied_3x3_areas: 已选墙块的所有3×3范围集合
            """
            # 终止条件：所有炸弹都选完 → 保存当前组合
            if bomb_index >= len(bomb_list):
                all_valid_combos.append(selected_pairs.copy())
                return

            # 当前要处理的炸弹
            current_bomb = bomb_list[bomb_index]
            # 当前炸弹的所有5×5可炸块
            current_walls = bomb_nearby_walls[current_bomb]

            # 遍历当前炸弹的每一个可炸块
            for wall in current_walls:
                # 计算当前墙块的3×3范围
                wall_3x3 = self._get_wall_occupied_area(wall)
                # 核心校验：当前墙块的3×3范围是否与已选墙块的3×3范围重叠
                if wall_3x3.isdisjoint(occupied_3x3_areas):
                    # 选中该墙块：加入已选配对 + 记录其3×3范围
                    selected_pairs.append((current_bomb, wall))
                    new_occupied = occupied_3x3_areas.union(wall_3x3)

                    # 处理下一个炸弹
                    backtrack(bomb_index + 1, selected_pairs, new_occupied)

                    # 回溯：移除当前墙块，继续遍历当前炸弹的下一个可炸块
                    selected_pairs.pop()

        # 从第一个炸弹开始遍历（初始：无已选配对、无已占用3×3范围）
        backtrack(bomb_index=0, selected_pairs=[], occupied_3x3_areas=set())

        return all_valid_combos

    # 依赖函数：计算单个墙块的3×3范围（复用你已有的）
    def _get_wall_occupied_area(self, wall_pos, radius=1):
        """获取单个墙块的3×3范围坐标集合"""
        wx, wy = wall_pos
        occupied = set()
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                check_x = wx + dx
                check_y = wy + dy
                # 确保坐标在地图内
                if 0 <= check_x < self.width and 0 <= check_y < self.height:
                    occupied.add((check_x, check_y))
        return occupied

    def investigate(self, ifbox=False, iftarget=False, order=0):
            """
            单次侦察一个目标（适配多次调用）
            :param ifbox: 是否侦察箱子（True/False，与iftarget二选一）
            :param iftarget: 是否侦察目标点（True/False，与ifbox二选一）
            :param order: 侦察目标的序号（从0开始，比如order=1表示第2个箱子/目标点）
            :return: 本次侦察的移动路径（方向列表），失败返回空列表
            """
            # # 1. 参数校验（二选一，且序号合法）
            # if (ifbox and iftarget) or (not ifbox and not iftarget):
            #     print("错误：ifbox和iftarget必须二选一")
            #     return []

            # 2. 确定本次侦察的目标点
            pre_path = []
            start_point = self.current_invest_pos
            pre_path.append(start_point)

            target_pos = None
            if ifbox:
                if order < 0 or order >= len(self.initial_boxes):
                    print(f"错误：箱子序号{order}超出范围（总数{len(self.initial_boxes)}）")
                    return []
                target_pos = self.initial_boxes[order]
                target_type = "箱子"
            else:
                if order < 0 or order >= len(self.initial_targets):
                    print(f"错误：目标点序号{order}超出范围（总数{len(self.initial_targets)}）")
                    return []
                target_pos = self.initial_targets[order]
                target_type = "目标点"

            print(f"\n开始侦察{target_type}：序号{order}，坐标{target_pos}")

            # 3. 找到目标点的可通行周边格子（非障碍、非其他元素）
            base_obstacles = set(self.walls)
            shortest_path = None
            min_path_len = float('inf')
            for b in self.initial_boxes:
                base_obstacles.add(b)
            for bo in self.initial_bombs:
                base_obstacles.add(bo)
            for t in self.initial_targets:
                base_obstacles.add(t)

            valid_surround_pos = self._find_valid_surround_pos(target_pos, base_obstacles)

            if not valid_surround_pos:
                print(f"{target_type}{target_pos}无可用周边格子，侦察失败")
                return None

            # 4. 智能体从当前位置移动到最近的可通行周边格子


            for surround_pos in valid_surround_pos:
                path = self._astar_move_to_pos(self.current_invest_pos,surround_pos,base_obstacles)
                if path and len(path) < min_path_len:
                    min_path_len = len(path)
                    shortest_path = path

            if not shortest_path:
                print(f"无法到达{target_type}{target_pos}的周边格子，侦察失败")
                return None

            # 5. 更新智能体当前位置（供下一次调用使用）
            sim_px, sim_py = self.current_invest_pos
            for dx, dy in shortest_path:
                sim_px += dx
                sim_py += dy

            pre_path.extend(self.actions_to_paths(self.current_invest_pos,shortest_path))
            self.current_invest_pos = (sim_px, sim_py)

            print(f"完成{target_type}{target_pos}侦察，本次移动步数：{len(shortest_path)}")
            print(f"智能体当前位置：{self.current_invest_pos}")

            return pre_path,target_pos

    def _find_valid_surround_pos(self, target_pos, obstacles):
        """
        找到目标点周边的可通行格子
        :param target_pos: 目标点坐标 (x,y)
        :param obstacles: 障碍物集合
        :return: 可通行格子列表 [(x1,y1), ...]
        """
        tx, ty = target_pos
        # 优先检查上下左右4个方向（最易站立的位置）
        dirs_4 = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        valid_pos = []

        for dx, dy in dirs_4:
            cx, cy = tx + dx, ty + dy
            # 校验条件：
            # 1. 在地图范围内 2. 不是墙壁 3. 不是箱子/目标点/炸弹（纯可站立空格）
            if (0 <= cx < self.width and 0 <= cy < self.height and
                    (cx, cy) not in obstacles):
                valid_pos.append((cx, cy))

        return valid_pos

    def _astar_move_to_pos(self, start_pos, end_pos, obstacles):
        """
        A*寻路：仅移动（不推箱子），从起点到终点
        :param start_pos: 智能体当前位置 (x,y)
        :param end_pos: 目标周边可通行格子 (x,y)
        :param obstacles: 障碍物集合（仅墙壁）
        :return: 移动路径（方向列表），None表示无解
        """
        dirs = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # 上下左右移动方向
        # 优先级队列：(总代价f, 已走步数g, 当前位置, 路径)
        pq = []
        # 启发函数h：曼哈顿距离
        h = abs(start_pos[0] - end_pos[0]) + abs(start_pos[1] - end_pos[1])
        heapq.heappush(pq, (h, 0, start_pos, []))

        visited = set()
        visited.add(start_pos)
        max_nodes = 500  # 防止寻路死循环

        while pq:
            f, g, (x, y), path = heapq.heappop(pq)

            # 到达终点，返回路径
            if (x, y) == end_pos:
                return path

            # 节点数超限，终止寻路
            if g > max_nodes:
                return None

            # 遍历所有移动方向
            for dx, dy in dirs:
                nx = x + dx
                ny = y + dy
                new_pos = (nx, ny)
                # 校验：地图内 + 非障碍 + 未访问
                if (0 <= nx < self.width and 0 <= ny < self.height and
                        new_pos not in obstacles and
                        new_pos not in visited):
                    visited.add(new_pos)
                    new_g = g + 1
                    new_h = abs(nx - end_pos[0]) + abs(ny - end_pos[1])
                    new_f = new_g + new_h  # f = g + h
                    heapq.heappush(pq, (new_f, new_g, new_pos, path + [(dx, dy)]))

        # 无可行路径
        return None

    def extract_turning_points(self,coord_list):
        """
        提取坐标列表中的拐点、起点、终点（删除平行路径的冗余坐标）
        :param coord_list: 连续坐标列表，如[(0,0), (0,1), (0,2), (1,2), (2,2)]
        :return: 处理后的关键坐标列表，如[(0,0), (0,2), (2,2)]
        """
        # 步骤1：输入校验
        if not isinstance(coord_list, list):
            print("错误：输入必须是列表格式！")
            return []
        if len(coord_list) <= 2:
            return coord_list  # 少于3个坐标，无需处理（保留全部）

        # 步骤2：初始化结果列表（先保留起点）
        result = [coord_list[0]]

        # 步骤3：遍历坐标，判断每三段的向量方向
        for i in range(1, len(coord_list)-1):
            # 获取连续三个点：前一个点(p_prev)、当前点(p_curr)、后一个点(p_next)
            p_prev = coord_list[i-1]
            p_curr = coord_list[i]
            p_next = coord_list[i+1]

            # 计算两段向量：v1 = p_curr - p_prev，v2 = p_next - p_curr
            v1_dx = p_curr[0] - p_prev[0]
            v1_dy = p_curr[1] - p_prev[1]
            v2_dx = p_next[0] - p_curr[0]
            v2_dy = p_next[1] - p_curr[1]

            # 判断向量是否平行（核心逻辑）：
            # 向量平行条件：v1_dx * v2_dy == v1_dy * v2_dx（叉乘为0）
            # 兼容浮点数：允许微小误差（1e-6）
            cross_product = v1_dx * v2_dy - v1_dy * v2_dx
            if abs(cross_product) > 1e-6:
                # 叉乘≠0 → 方向变化，当前点是拐点，保留
                result.append(p_curr)

        # 步骤4：添加终点（必须保留）
        result.append(coord_list[-1])

        return result

    def actions_to_paths(self,initial_pos, action_list, step=1):
        """
        将动作列表转换为完整的坐标路径列表
        :param initial_pos: 初始位置元组，如(0, 0)（x, y）
        :param action_list: 动作列表，如['上', '右', '下', '左']，支持的动作：上/下/左/右
        :param step: 每个动作的步长（默认1，即每次移动1个格子）
        :return: 完整的坐标路径列表，如[(0,0), (0,-1), (1,-1), (1,0), (0,0)]
        """
        # 步骤1：输入校验
        if not isinstance(initial_pos, (tuple, list)) or len(initial_pos) != 2:
            print("错误：初始位置必须是(x,y)格式的元组/列表！")
            return []
        if not isinstance(action_list, list) or len(action_list) == 0:
            print("错误：动作列表必须是非空列表！")
            return [tuple(initial_pos)]  # 无动作时仅返回初始位置

        # 步骤2：定义动作对应的坐标偏移（可根据你的坐标系调整）
        # 坐标系规则：右=x+1，左=x-1，上=y-1，下=y+1（屏幕/网格通用规则）
        # action_offset = {
        #     "右": (step, 0),
        #     "左": (-step, 0),
        #     "上": (0, -step),
        #     "下": (0, step)
        # }

        # 步骤3：初始化路径列表（先加入初始位置）
        path = [tuple(initial_pos)]  # 确保初始位置是元组，避免后续修改影响
        current_x, current_y = initial_pos  # 当前坐标

        # 步骤4：遍历动作列表，生成路径
        for action in action_list:
            # 获取动作对应的偏移量
            dx, dy = action[0], action[1]
            # 更新当前坐标
            current_x += dx
            current_y += dy
            # 记录新坐标到路径
            path.append((current_x, current_y))

        return path

    def permutations(self,iterable, r=None):
        """
        纯Python实现itertools.permutations功能
        :param iterable: 可迭代对象（如range(n)、列表、元组等）
        :param r: 排列长度，默认None（生成全排列）
        :return: 排列生成器（可转为列表），每个元素是元组形式的排列
        """
        pool = tuple(iterable)
        n = len(pool)
        r = n if r is None else r
        if r > n:
            return []

        def _permute(elements, length):
            if length == 1:
                return [(elem,) for elem in elements]

            result = []
            for i in range(len(elements)):
                current = elements[i]
                rest = elements[:i] + elements[i+1:]
                for p in _permute(rest, length - 1):
                    result.append((current,) + p)
            return result
        return _permute(pool, r)

