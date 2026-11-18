"""One bounded vision judge request; scoring and gold stay in evaluation."""
import base64

from harness.core.budget import Budget, output_text
from harness.evidence.archive import InvocationArchive, now
from harness.llm import Responses


def judge(context, frames, prompt):
    archive=InvocationArchive(context.directory,role='evaluation_vqa',task_id=context.id)
    config={**context.config,'critic_model':'kimi-k3',
            'request_timeout_s':context.config.get('evaluation',{}).get('vqa_timeout_s',1000)}
    client=Responses(archive,config,context.services.store.api_key(),budget=Budget(1,None),
        stopped=context.stopped,before_call=lambda:context.services.app.verify_runtime(context.record))
    content=[{'type':'input_image','image_url':'data:image/jpeg;base64,'+
        base64.b64encode(path.read_bytes()).decode()} for path in frames]
    content.append({'type':'input_text','text':prompt})
    try:
        response=client.call('',[{'role':'user','content':content}],request_kind='evaluation_vqa',temperature=0)
        archive.update(status='completed',finished_at=now(),usage=archive.usage())
        return output_text(response)
    except Exception as exc:
        archive.update(status='interrupted' if isinstance(exc,InterruptedError) else 'failed',error=str(exc),finished_at=now())
        raise
