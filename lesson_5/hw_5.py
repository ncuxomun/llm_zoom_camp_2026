from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter, 
    SpanExportResult,
)

import sqlite3

from starter import index, client
from rag_helper import RAGBase

# To store traces
class SQLiteSpanExporter(SpanExporter):

    def __init__(self, db_path="traces_q6.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,
                start_time INTEGER,
                end_time INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL
            )
        """)
        self.conn.commit()

    def export(self, spans):
        for span in spans:
            attrs = dict(span.attributes or {})
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.conn.close()

    def force_flush(self):
        return True


provider = TracerProvider()

# # Console span processor for Q1-Q3
# provider.add_span_processor(
#     SimpleSpanProcessor(ConsoleSpanExporter())
# )

# Console span processor for Q4-
provider.add_span_processor(
    SimpleSpanProcessor(SQLiteSpanExporter("traces.db"))
)

trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-zoomcamp")


class RAGTraced(RAGBase):

    def search(self, query, num_results=5):
        with tracer.start_as_current_span("search"):
            return super().search(query, num_results)

    def llm(self, prompt):
        with tracer.start_as_current_span("llm") as span:
        # with tracer.start_as_current_span("llm"):
            response = super().llm(prompt)
            usage = response.usage
            span.set_attribute("input_tokens", usage.input_tokens)
            span.set_attribute("output_tokens", usage.output_tokens)

            # return super().llm(prompt)
            return response
            

    def rag(self, query):
        with tracer.start_as_current_span("rag"):
            return super().rag(query)


rag = RAGTraced(index=index, llm_client=client)

query = "How does the agentic loop keep calling the model until it stops?"
# answer = rag.rag(query)
# print(answer)


#####* Answers
## Q1: 3 (1 for each - search, llm, and rag)
## Q2: 7111 or 7000
"""   "name": "llm",
    "context": {
        "trace_id": "0xc2d37db59f734a5aa6f1d0899f1f981c",
        "span_id": "0x222c4ee96bc3660b",
        "trace_state": "[]"
    },
    "kind": "SpanKind.INTERNAL",
    "parent_id": "0x9506dd3f57136d2e",
    "start_time": "2026-07-20T00:42:58.660949Z",
    "end_time": "2026-07-20T00:43:01.069331Z",
    "status": {
        "status_code": "UNSET"
    },
    "attributes": {
        "input_tokens": 7111, #!
        "output_tokens": 170
    },
    "events": [],
    "links": [],
    "resource": {
        "attributes": {
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.version": "1.44.0",
            "service.instance.id": "f368e48e-a909-446a-b270-ada61e85a3e1",
            "service.name": "unknown_service"
        },
        "schema_url": ""
    }
}
"""
## Q3: ~2+ secs or 100-500ms
"""
"start_time": "2026-07-20T00:46:45.595800Z",
"end_time": "2026-07-20T00:46:47.947443Z",
"""

## Q4: search, llm, rag
"""
## To check 
conn = sqlite3.connect('lesson_5/traces.db')

for row in conn.execute('SELECT DISTINCT name FROM spans'):
    print(row[0])
"""
## Q5:
# conn = sqlite3.connect('lesson_5/traces.db')

# for i, row in enumerate(conn.execute('SELECT * FROM spans')):
#     if row[0] != "rag":
#         print(f"Row #{i+1} // Task #{row[0]} // Duration #{float(row[2] - row[1])}")
#     # print(row)
    # break
""" #* Naive way but the outcomes are quite obvious - it is the 'llm' that is the time consuming
Row #1 // Task #search // Duration #913722.0
Row #2 // Task #llm // Duration #2517482285.0
Row #4 // Task #search // Duration #829001.0
Row #5 // Task #llm // Duration #2264459070.0
Row #7 // Task #search // Duration #882971.0
Row #8 // Task #llm // Duration #2492942860.0
Row #10 // Task #search // Duration #929530.0
Row #11 // Task #llm // Duration #2611503663.0
Row #13 // Task #search // Duration #859158.0
Row #14 // Task #llm // Duration #2123716777.0
Row #16 // Task #search // Duration #898969.0
Row #17 // Task #llm // Duration #3295707841.0
Row #19 // Task #search // Duration #933569.0
Row #20 // Task #llm // Duration #2420492071.0
"""

## Q6: Identical
# import pandas as pd

# row = []
# conn = sqlite3.connect('traces_q6.db')

# df = pd.read_sql("""
#        SELECT name, start_time, end_time, input_tokens
#        FROM spans ORDER BY rowid ASC
#    """, conn)

# print(df)
"""
      name           start_time             end_time  input_tokens
0   search  1784510382150417108  1784510382151342430           NaN
1      llm  1784510382158955048  1784510384396261191        7111.0
2      rag  1784510382150386788  1784510384412976611           NaN
3   search  1784510385860172159  1784510385861122333           NaN
4      llm  1784510385878061683  1784510388627587362        7111.0
5      rag  1784510385860141267  1784510388636208870           NaN
6   search  1784510390080210088  1784510390081112873           NaN
7      llm  1784510390097524707  1784510391930175330        7111.0
8      rag  1784510390080181851  1784510391946876985           NaN
9   search  1784510393339581674  1784510393340438171           NaN
10     llm  1784510393357654967  1784510395299873517        7111.0
11     rag  1784510393339551362  1784510395307629710           NaN
"""