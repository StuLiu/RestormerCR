# 决赛选手提前准备docker
export URL="registry.cn-hangzhou.aliyuncs.com/liuwa/cloud_removal"
export USERNAMEPC="liuwang20144623"
export PASSWORD="qaz520520"
export TEAMID="00f0ecc1-019b-4d49-8a62-65d14d413641"
# 用这个命令直接提交这个一运行 就会自动停止并删除上一次运行的container
# tag 必须是 juesai  不能是其他的 后台会自动根据 juesai这个tag 去pull
curl -X POST -H "Content-Type: application/json" \
  -d '{"url":"'$URL'", "tag":"juesai", "username":"'$USERNAMEPC'", "password":"'$PASSWORD'", "teamid":"'$TEAMID'"}' \
  http://119.3.123.24:6001/docker

# curl执行后会返回jupyterlab的公网端口号 可以自行去测试jupyterlab运行情况
curl -X POST -H "Content-Type: application/json" \
  -d '{"teamid":"'$TEAMID'"}' \
  http://119.3.123.24:6002/logs