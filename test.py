from testmd.scanner_contorl import *
from testmd.vor_graph_generation import *
from testmd.scanner_control_with_vor_graph import *
from testmd.gym_env import *
from testmd.test_model import *

if __name__ == "__main__":
    #test_scanner_control()
    #test_vor_graph_generation()
    #test_scanner_control_with_vor_graph()
    #test_env_wrapper()
    test_model("a2c_2e6")
    pass
