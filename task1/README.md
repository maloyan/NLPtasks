# Малоян Нарек, 427
Запускаете Maloyan.py, он выведет файлы с разметкой в папку ./result

Затем можно запустить скрипт проверки результата:
```python
python scripts/t1_eval.py -s ./testset -t ./result
```

Могут возникнуть проблемы (маловероятно, но на всякий случай) с
```python
import util
```
Тогда надо вручную добавить в PATH:
 ```bash
 export PATH=$HOME/NLPTasks/task1/scripts/dialent/task1/:$HOME/NLPTasks/task1/scripts/dialent/:$HOME/NLPTasks/task1/scripts/:$PATH
 ```
