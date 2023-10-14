from langchain.schema import AIMessage, HumanMessage
import gradio as gr
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from threading import Thread
from myllm import MyLLM, PrintCall


llm = MyLLM(stream=True, ali_model=True)

chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template("{ques}"),
    # todo fix
    # 这种方式传callback未生效，下面那种方式生效
    # callbacks=[PrintCall()]
)


def local_test():
    # 直接模型调用
    for i in llm.stream("你是谁"):
        print(i)
    # chain 方式调用,callback只有开启stream才有效果
    res = chain("test", callbacks=[PrintCall()])
    print(res)


def QA(message, history):
    messages = []
    for msg in history:
        messages.append(HumanMessage(content=msg[0]))
        messages.append(AIMessage(content=msg[1]))
    messages.append(HumanMessage(content=message))
    cal = PrintCall(print=False)

    def work():
        chain(message, callbacks=[cal])

    # 因为chain调用是阻塞的，所以放到线程去调用
    t = Thread(target=work)
    t.start()
    # 处理结果返回中每次回调
    while True:
        t = cal.next()
        if t == cal.Flag.End:
            break
        yield t


local_test()

gr.ChatInterface(QA).queue().launch(server_port=8888, server_name="0.0.0.0")
