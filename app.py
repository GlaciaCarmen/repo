from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import random
import os
import pandas as pd
import numpy as np
from flask import jsonify, request
from water_quality_visualization import load_all_water_data

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 可自行修改

# 加载水质数据（在应用启动时）
water_data_path = os.path.join('软件工程大作业数据', '水质数据')
all_water_data = load_all_water_data(water_data_path)
if not isinstance(all_water_data, pd.DataFrame) or all_water_data.empty:
    all_water_data = pd.DataFrame()
    print("警告：水质数据加载失败，使用空数据集")

# 添加调试信息
print(f"当前工作目录: {os.getcwd()}")
print(f"数据目录存在性: {os.path.exists(water_data_path)}")

# 加载数据并处理异常
try:
    all_water_data = load_all_water_data(water_data_path)
except Exception as e:
    print(f"数据加载异常: {str(e)}")
    all_water_data = pd.DataFrame()

# 空数据保护
if not isinstance(all_water_data, pd.DataFrame) or all_water_data.empty:
    all_water_data = pd.DataFrame()
    print("警告：水质数据为空，相关功能将受限")

# 获取监测点列表
monitoring_sites = []
if not all_water_data.empty:
    monitoring_sites = pd.concat([
        all_water_data.get('数据来源_监测点', pd.Series(dtype='object')),
        all_water_data.get('断面名称', pd.Series(dtype='object'))
    ]).dropna().unique().tolist()

water_parameters = [
    '水温(℃)', 'pH(无量纲)', '溶解氧(mg/L)', '电导率(μS/cm)',
    '浊度(NTU)', '高锰酸盐指数(mg/L)', '氨氮(mg/L)',
    '总磷(mg/L)', '总氮(mg/L)', '叶绿素α(mg/L)', '藻密度(cells/L)'
]

# 添加空值处理
all_water_data['数据来源_监测点'] = all_water_data['数据来源_监测点'].fillna('默认监测点')

@app.route('/api/water_quality')
def water_quality_data():
    site = request.args.get('site')
    param = request.args.get('param')

    try:
        # 同时匹配监测点和断面名称
        query = (
                (all_water_data['数据来源_监测点'] == site) |
                (all_water_data['断面名称'] == site)
        )
        site_data = all_water_data[query].sort_values('监测时间')

        # 处理重复时间戳
        site_data = site_data.drop_duplicates('监测时间')

        # 处理异常值
        numeric_cols = ['水温(℃)', 'pH(无量纲)', '溶解氧(mg/L)', '电导率(μS/cm)',
                        '浊度(NTU)', '高锰酸盐指数(mg/L)', '氨氮(mg/L)',
                        '总磷(mg/L)', '总氮(mg/L)']
        site_data[numeric_cols] = site_data[numeric_cols].replace(['*', '--'], np.nan)

        # 生成时间序列
        time_series = site_data['监测时间'].dt.strftime('%Y-%m-%dT%H:%MZ').tolist()
        values = site_data[param].interpolate().ffill().bfill().tolist()

        return jsonify({
            'time': time_series,
            'values': values,
            'unit': get_unit(param),
            'warnings': detect_anomalies(site_data)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_hydrology_data():
    """获取最新水文数据"""
    latest = all_water_data.sort_values('监测时间').iloc[-1]
    return {
        'voltage': round(random.uniform(3.2, 3.4), 1),  # 模拟电压数据
        'temperature': latest['水温(℃)'],
        'salinity': latest['电导率(μS/cm)'],
        'do': latest['溶解氧(mg/L)'],
        'update_time': latest['监测时间'].strftime('%Y-%m-%d %H:%M')
    }

@app.route('/api/hydrology')
def hydrology_data():
    return jsonify(get_hydrology_data())

# 加载鱼类数据
# 修改原始数据加载部分
fish_csv_path = os.path.join('软件工程大作业数据', 'Fish.csv')
df_fish = pd.read_csv(fish_csv_path)
fish_species = df_fish['Species'].unique().tolist()
fish_variables = [col for col in df_fish.columns if col != 'Species']

# 新增鱼类数据分析函数
def analyze_fish_data():
    df = pd.read_csv('软件工程大作业数据/Fish.csv')

    # 计算各鱼种统计数据
    fish_stats = {}
    for species in df['Species'].unique():
        species_df = df[df['Species'] == species]
        fish_stats[species] = {
            'total': len(species_df),
            'weight_stats': species_df['Weight(g)'].describe(),
            'length_stats': species_df['Length1(cm)'].describe()
        }

    # 生成生长趋势数据（假设时间序列）
    dates = pd.date_range(end=pd.Timestamp.today(), periods=12, freq='ME').strftime('%Y-%m').tolist()
    growth_values = np.random.randint(1000, 2000, size=12).tolist()  # 实际应根据真实数据生成

    return {
        'species_stats': fish_stats,
        'growth_trend': {'dates': dates, 'values': growth_values},
        'all_data': df
    }


# 初始化时加载数据
fish_analysis = analyze_fish_data()


@app.route('/api/fish_distribution')
def fish_distribution_data():
    species = request.args.get('species')
    var = request.args.get('var')

    if species not in fish_species or var not in fish_variables:
        return jsonify({'error': 'Invalid parameters'}), 400

    species_data = df_fish[df_fish['Species'] == species]

    # 自动计算最佳分箱
    if var == 'Weight(g)':
        max_weight = species_data[var].max()
        bins = list(range(0, int(max_weight) + 100, 100))
    else:
        bins = 10

    counts, bins = np.histogram(species_data[var], bins=bins)
    bin_labels = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

    return jsonify({
        'bins': bin_labels,
        'counts': counts.tolist()
    })

# ---------- 数据库初始化 ----------
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user'
        )
    ''')
    conn.commit()
    conn.close()

# ---------- 首页 ----------
@app.route('/')
def index():
    return render_template('home.html')

# ---------- 注册 ----------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        role = request.form['role']
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', (username, password, role))
            conn.commit()
            flash("注册成功，请登录", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("用户名已存在", "danger")
        finally:
            conn.close()
    return render_template('register.html')

# ---------- 登录 ----------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password_input = request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username=?', (username,))
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password_input):
            session['username'] = username
            session['role'] = user[3]
            flash("登录成功", "success")
            if user[3] == 'admin':
                return redirect(url_for('admin_home'))
            elif user[3] == 'farmer':
                return redirect(url_for('farmer_home'))
            else:
                return redirect(url_for('user_home'))
        else:
            flash("用户名或密码错误", "danger")
    return render_template('login.html')

# ---------- 用户主页 ----------
@app.route('/user_home')
def user_home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('user_home.html')

# ---------- 水下系统 ----------
@app.route('/underwater')
def underwater():
    # 从水质数据获取最新环境数据
    latest_water = all_water_data.sort_values('监测时间').iloc[-1]
    environment_data = {
        'temperature': latest_water['水温(℃)'],
        'salinity': latest_water['电导率(μS/cm)'],  # 根据实际数据调整
        'oxygen': latest_water['溶解氧(mg/L)'],
        'score': round(latest_water[['pH(无量纲)', '溶解氧(mg/L)']].mean() * 10, 1)
    }

    # 从鱼类数据生成统计信息
    total_fish = len(fish_analysis['all_data'])
    species_count = len(fish_analysis['species_stats'])

    species_distribution = [
        {'value': stats['total'], 'name': species}
        for species, stats in fish_analysis['species_stats'].items()
    ]

    fish_data = {
        'total': total_fish,
        'species': species_count,
        'seedlings': fish_analysis['all_data']['Weight(g)'].lt(100).sum(),  # 假设重量<100g为鱼苗
        'current_growth': fish_analysis['all_data']['Weight(g)'].gt(500).sum(),
        'species_distribution': species_distribution
    }

    # 属性数据从鱼类数据计算
    attributes_data = {
        'weight': fish_analysis['all_data']['Weight(g)'].tolist(),
        'size': fish_analysis['all_data']['Length1(cm)'].tolist(),
        'lifespan': np.random.randint(1, 5, total_fish).tolist()  # 示例数据，需根据实际数据调整
    }

    hardware_data = {
        'cages': 10,
        'cameras': 8,
        'sonars': 3,
        'sensors': {
            'total': 20,
            'online': 18,
            'maintenance': 2
        }
    }

    return render_template('underwater.html',
                           data={
                               'environment': environment_data,
                               'fish': fish_data,
                               'hardware': hardware_data,  # 添加硬件数据
                               'attributes': attributes_data
                           },
                           fish_species=fish_species,
                           fish_variables=fish_variables,
                           dates=fish_analysis['growth_trend']['dates'],
                           values=fish_analysis['growth_trend']['values']
                           )

# ---------- 主要信息 ----------
@app.route('/main_info')
def main_info():
    # 从水质数据获取最新水文数据
    latest_water = all_water_data.sort_values('监测时间').iloc[-1]
    hydrology_data = {
        'voltage': 3.3,  # 需要实际传感器数据
        'temperature': latest_water['水温(℃)'],
        'salinity': latest_water['电导率(μS/cm)'],
        'do': latest_water['溶解氧(mg/L)']
    }

    # 生成摄像头位置数据（示例）
    cameras_data = [
        {'id': 1, 'position': [36.07, 120.38], 'status': 'online'},
        {'id': 2, 'position': [36.08, 120.39], 'status': 'offline'}
    ]

    return render_template('main_info.html',
                          hydrology=hydrology_data,
                          cameras=cameras_data)

# ---------- 智能中心 ----------
# 添加视频流生成函数和路由
def generate_frames(camera_id=0):
    camera = cv2.VideoCapture(camera_id)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """视频流路由"""
    camera_id = request.args.get('camera', 0, type=int)
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/intelligence')
def intelligence():
    # 从水质数据获取最新环境数据
    latest_water = all_water_data.sort_values('监测时间').iloc[-1]
    env_data = {
        'temp': latest_water['水温(℃)'],
        'salinity': latest_water['电导率(μS/cm)'],
        'do': latest_water['溶解氧(mg/L)'],
        'ph': latest_water['pH(无量纲)']
    }

    # 计算环境评分
    env_score = round(latest_water[['溶解氧(mg/L)', 'pH(无量纲)', '水温(℃)']].mean() * 10, 1)

    return render_template('intelligence.html',
                           env_data=env_data,
                           env_score=env_score,
                           monitoring_sites=monitoring_sites,
                           water_parameters=water_parameters)
# ---------- 管理员主页 ----------
@app.route('/admin_home')
def admin_home():
    if session.get('role') != 'admin':
        return "无权限", 403
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, username, role FROM users")
    users = c.fetchall()
    conn.close()
    return render_template('manage_users.html', users=users)

# 用户管理（仅管理员可访问）
@app.route('/admin/users')
def manage_users():
    if session.get('role') != 'admin':
        return "无权限", 403
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, username, role FROM users")
    users = c.fetchall()
    conn.close()
    return render_template('manage_users.html', users=users)

# 删除用户（仅管理员可操作，禁止删除管理员）
@app.route('/admin/delete_user/<int:user_id>')
def delete_user(user_id):
    if session.get('role') != 'admin':
        return "无权限", 403
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE id = ?", (user_id,))
    role = c.fetchone()
    if role and role[0] != 'admin':
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    conn.close()
    return redirect(url_for('manage_users'))


@app.route('/farmer_home')
def farmer_home():
    if session.get('role') != 'farmer':
        flash("无权限访问养殖户页面", "danger")
        return redirect(url_for('index'))
    return render_template('farmer_home.html')


# ---------- 退出登录 ----------
@app.route('/logout')
def logout():
    session.clear()
    flash("您已退出", "info")
    return redirect(url_for('login'))

# ---------- 主程序启动 ----------
if __name__ == '__main__':
    init_db()
    app.run(debug=True)