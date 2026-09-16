"""Loopback OpenAI-compatible endpoint for official GUI benchmark clients.

Non-streaming only, serial GPU inference, no outbound image downloads. Official
benchmark prompts/action formatting are preserved. This server does NOT judge
its own task outcomes. Use separate evaluator credentials in MemGUI-Bench.
"""
import argparse,json,time,threading,traceback
from http.server import BaseHTTPRequestHandler,HTTPServer
import torch
from .qwen import QwenEngine

class Server(HTTPServer):
    engine=None;controlled=False;request_seed=0;adapter=None
class Handler(BaseHTTPRequestHandler):
    def response(self,code,payload):
        data=json.dumps(payload).encode();self.send_response(code);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_GET(self):
        if self.path=='/health':return self.response(200,{'status':'ok','controlled':self.server.controlled,'model':self.server.engine.model_path})
        if self.path in {'/v1/models','/models'}:return self.response(200,{'object':'list','data':[{'id':'tango-gui','object':'model'}]})
        return self.response(404,{'error':'not found'})
    def do_POST(self):
        if self.path not in {'/v1/chat/completions','/chat/completions'}:return self.response(404,{'error':'not found'})
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=64*1024*1024:raise ValueError('Invalid request size')
            p=json.loads(self.rfile.read(size))
            if p.get('stream'):raise ValueError('Streaming is not implemented; set stream=false')
            torch.manual_seed(int(p.get('seed',self.server.request_seed)))
            answer=self.server.engine.generate({'messages':p['messages']},self.server.controlled,
                float(p.get('temperature',0)),int(p.get('max_tokens',256)))
            self.response(200,{'id':f'tango-{time.time_ns()}','object':'chat.completion','created':int(time.time()),
                'model':'tango-gui','choices':[{'index':0,'message':{'role':'assistant','content':answer},'finish_reason':'stop'}]})
        except Exception as exc:
            traceback.print_exc();self.response(400,{'error':{'message':str(exc),'type':'invalid_request_error'}})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint');p.add_argument('--model')
    p.add_argument('--adapter');p.add_argument('--device',default='cuda');p.add_argument('--port',type=int,default=8127)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    if not a.checkpoint and not a.model:raise ValueError('Provide --model or --checkpoint')
    engine=QwenEngine.from_checkpoint(a.checkpoint,a.device,a.model) if a.checkpoint else QwenEngine(a.model,device=a.device)
    if a.adapter:
        if a.checkpoint:raise ValueError('No-hook student server must not also apply TANGO teacher checkpoint')
        engine.model.load_adapter(a.adapter)
    server=Server((a.host,a.port),Handler);server.engine=engine;server.controlled=bool(a.checkpoint);server.request_seed=a.seed
    print(json.dumps({'host':a.host,'port':a.port,'ready':True}),flush=True)
    try:server.serve_forever()
    finally:server.server_close()
if __name__=='__main__':main()
