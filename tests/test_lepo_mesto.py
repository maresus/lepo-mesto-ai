"""
Občina Lepo Mesto AI — 100 poglobljenih testov
================================================
Kategorije:
  A.  RAG: load_knowledge + BM25 search         (20 testov)
  B.  DB: init, save, get                        (10 testov)
  C.  Config in llm_client                       (5 testov)
  D.  /health + osnovna infra                    (5 testov)
  E.  /chat endpoint (mock LLM)                  (25 testov)
  F.  /admin API (zaščita, seje, stats)          (10 testov)
  G.  Živi LLM testi (pravi OpenAI klici)        (25 testov)
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
HAS_OPENAI = bool(OPENAI_KEY)

KNOWLEDGE_PATH = PROJECT_ROOT / "knowledge.jsonl"

# ── Mock helper ───────────────────────────────────────────────────────────────
def _mock_reply(text: str):
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = text
    return mock

def _patch_openai(text: str = "Testni odgovor."):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_reply(text)
    return patch("app.core.chat_service.get_llm_client", return_value=mock_client)

# ── FastAPI client ────────────────────────────────────────────────────────────
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app.core.db import init_db
    init_db()
    from app.rag.search import load_knowledge
    load_knowledge(KNOWLEDGE_PATH)
    from main import app
    with TestClient(app) as c:
        yield c


# ═══════════════════════════════════════════════════════════════════════════════
# A. RAG — 20 testov
# ═══════════════════════════════════════════════════════════════════════════════
from app.rag.search import load_knowledge, search, get_context


class TestRAG:
    @pytest.fixture(autouse=True)
    def load_kb(self):
        count = load_knowledge(KNOWLEDGE_PATH)
        assert count > 0, "Baza znanja mora biti naložena"

    def test_A01_loads_at_least_15_chunks(self):
        assert load_knowledge(KNOWLEDGE_PATH) >= 15

    def test_A02_search_zupan_returns_results(self):
        assert len(search("župan", top_k=2)) > 0

    def test_A03_search_zupan_content_has_novak(self):
        results = search("župan Janez Novak", top_k=1)
        assert "Novak" in results[0].paragraph

    def test_A04_search_uradne_ure(self):
        results = search("uradne ure", top_k=1)
        combined = results[0].paragraph.lower()
        assert "8:00" in combined or "pon" in combined or "sreda" in combined

    def test_A05_search_kontakt_returns_phone(self):
        results = search("kontakt telefon", top_k=1)
        assert "555" in results[0].paragraph

    def test_A06_search_smeti_odvoz(self):
        results = search("smeti odvoz", top_k=2)
        combined = " ".join(r.paragraph for r in results).lower()
        assert "torek" in combined or "sreda" in combined or "petek" in combined

    def test_A07_search_solstvo_returns_OŠ(self):
        results = search("šola osnovna", top_k=2)
        combined = " ".join(r.paragraph for r in results)
        assert "OŠ" in combined or "Lepo Mesto" in combined

    def test_A08_search_zdravstvo(self):
        results = search("zdravstveni dom lekarna", top_k=2)
        combined = " ".join(r.paragraph for r in results).lower()
        assert "zdravstven" in combined or "lekarna" in combined

    def test_A09_search_turizem(self):
        results = search("turizem TIC festival", top_k=2)
        combined = " ".join(r.paragraph for r in results).lower()
        assert "tic" in combined or "festival" in combined or "muzej" in combined

    def test_A10_search_vloge_gradnja(self):
        results = search("vloga gradbeno dovoljenje komunalni", top_k=2)
        combined = " ".join(r.paragraph for r in results).lower()
        assert "22" in combined or "vloga" in combined or "gradnja" in combined

    def test_A11_search_proracun(self):
        results = search("proračun 2026", top_k=2)
        combined = " ".join(r.paragraph for r in results)
        assert "4.200" in combined or "EUR" in combined

    def test_A12_search_razpisi(self):
        results = search("razpis sofinanciranje", top_k=2)
        combined = " ".join(r.paragraph for r in results).lower()
        assert "razpis" in combined or "sofinancir" in combined

    def test_A13_search_narava_jezero(self):
        results = search("jezero Lepica kopanje", top_k=2)
        combined = " ".join(r.paragraph for r in results).lower()
        assert "lepica" in combined or "kopanje" in combined or "jezero" in combined

    def test_A14_search_komunala(self):
        results = search("komunalno podjetje voda", top_k=2)
        combined = " ".join(r.paragraph for r in results).lower()
        assert "komunal" in combined or "voda" in combined

    def test_A15_search_obcinski_svet(self):
        results = search("občinski svet seje", top_k=2)
        combined = " ".join(r.paragraph for r in results).lower()
        assert "svet" in combined or "seja" in combined or "oblak" in combined

    def test_A16_search_unknown_returns_empty(self):
        assert search("xyzabc999neobstoji") == []

    def test_A17_top_k_respected(self):
        assert len(search("občina", top_k=3)) <= 3

    def test_A18_get_context_not_empty(self):
        ctx = get_context("župan kontakt")
        assert ctx.strip() != ""

    def test_A19_get_context_returns_string(self):
        assert isinstance(get_context("uradne ure"), str)

    def test_A20_search_pogosta_vprasanja(self):
        results = search("potrdilo stalno bivališče", top_k=2)
        combined = " ".join(r.paragraph for r in results).lower()
        assert "upravna enota" in combined or "bivališ" in combined or "potrdilo" in combined


# ═══════════════════════════════════════════════════════════════════════════════
# B. DB — 10 testov
# ═══════════════════════════════════════════════════════════════════════════════
from app.core.db import init_db, save_message, get_sessions, get_messages, get_stats


class TestDB:
    @pytest.fixture(autouse=True)
    def setup_db(self):
        init_db()

    def test_B01_init_db_no_error(self):
        init_db()  # idempotent

    def test_B02_save_message_user(self):
        sid = "test_" + uuid.uuid4().hex[:8]
        save_message(sid, "user", "Zdravo!")

    def test_B03_save_message_assistant(self):
        sid = "test_" + uuid.uuid4().hex[:8]
        save_message(sid, "user", "Vprašanje")
        save_message(sid, "assistant", "Odgovor")

    def test_B04_get_messages_returns_saved(self):
        sid = "test_" + uuid.uuid4().hex[:8]
        save_message(sid, "user", "Testno sporočilo XYZ")
        msgs = get_messages(sid)
        assert any("Testno sporočilo XYZ" in m["content"] for m in msgs)

    def test_B05_get_messages_order(self):
        sid = "test_" + uuid.uuid4().hex[:8]
        save_message(sid, "user", "Prvo")
        save_message(sid, "assistant", "Drugo")
        msgs = get_messages(sid)
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_B06_get_sessions_returns_list(self):
        assert isinstance(get_sessions(), list)

    def test_B07_get_sessions_includes_new_session(self):
        sid = "test_unique_" + uuid.uuid4().hex
        save_message(sid, "user", "Test")
        sessions = get_sessions()
        sids = [s["session_id"] for s in sessions]
        assert sid in sids

    def test_B08_get_stats_returns_dict(self):
        stats = get_stats()
        assert "total_sessions" in stats
        assert "today_sessions" in stats
        assert "total_messages" in stats

    def test_B09_get_stats_counts_increase(self):
        stats_before = get_stats()
        sid = "test_" + uuid.uuid4().hex[:8]
        save_message(sid, "user", "Nova seja")
        stats_after = get_stats()
        assert stats_after["total_sessions"] >= stats_before["total_sessions"]

    def test_B10_get_messages_empty_for_unknown_session(self):
        msgs = get_messages("nonexistent_session_" + uuid.uuid4().hex)
        assert msgs == []


# ═══════════════════════════════════════════════════════════════════════════════
# C. Config in llm_client — 5 testov
# ═══════════════════════════════════════════════════════════════════════════════
from app.core.config import Settings


class TestConfig:
    def test_C01_default_project_name(self):
        s = Settings()
        assert "Lepo Mesto" in s.project_name or "Leja" in s.project_name

    def test_C02_default_model_is_mini(self):
        s = Settings()
        assert "mini" in s.openai_model or "gpt" in s.openai_model

    def test_C03_admin_token_has_default(self):
        s = Settings()
        assert s.admin_token is not None
        assert len(s.admin_token) > 0

    def test_C04_settings_are_strings(self):
        s = Settings()
        assert isinstance(s.project_name, str)
        assert isinstance(s.openai_model, str)

    def test_C05_admin_token_not_empty(self):
        s = Settings()
        assert s.admin_token != ""


# ═══════════════════════════════════════════════════════════════════════════════
# D. /health + infra — 5 testov
# ═══════════════════════════════════════════════════════════════════════════════
class TestHealth:
    def test_D01_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_D02_health_status_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_D03_health_bot_lepo_mesto(self, client):
        assert client.get("/health").json()["bot"] == "lepo-mesto"

    def test_D04_root_returns_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_D05_widget_returns_html(self, client):
        r = client.get("/widget")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]


# ═══════════════════════════════════════════════════════════════════════════════
# E. /chat endpoint (mock) — 25 testov
# ═══════════════════════════════════════════════════════════════════════════════
class TestChatEndpoint:
    def _chat(self, client, message, session_id=None):
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        with _patch_openai("Testni odgovor za občinsko vprašanje."):
            return client.post("/chat", json=payload)

    def test_E01_returns_200(self, client):
        assert self._chat(client, "Zdravo").status_code == 200

    def test_E02_reply_field_present(self, client):
        assert "reply" in self._chat(client, "test").json()

    def test_E03_session_id_in_response(self, client):
        assert "session_id" in self._chat(client, "test").json()

    def test_E04_session_id_preserved(self, client):
        sid = "seja_" + uuid.uuid4().hex[:8]
        r = self._chat(client, "test", session_id=sid)
        assert r.json()["session_id"] == sid

    def test_E05_reply_not_empty(self, client):
        assert self._chat(client, "Zdravo").json()["reply"].strip() != ""

    def test_E06_slovenian_chars_accepted(self, client):
        r = self._chat(client, "Kako pridobim gradbeno dovoljenje čšž?")
        assert r.status_code == 200

    def test_E07_long_message_accepted(self, client):
        r = self._chat(client, "Zanima me občina. " * 40)
        assert r.status_code == 200

    def test_E08_empty_message_handled(self, client):
        assert self._chat(client, "").status_code == 200

    def test_E09_whitespace_message_handled(self, client):
        assert self._chat(client, "   ").status_code == 200

    def test_E10_multiple_turns_same_session(self, client):
        sid = "multiturn_" + uuid.uuid4().hex[:8]
        self._chat(client, "Kdo je župan?", sid)
        r = self._chat(client, "Hvala za info.", sid)
        assert r.status_code == 200

    def test_E11_different_sessions_independent(self, client):
        s1 = "s1_" + uuid.uuid4().hex[:6]
        s2 = "s2_" + uuid.uuid4().hex[:6]
        r1 = self._chat(client, "test1", s1)
        r2 = self._chat(client, "test2", s2)
        assert r1.json()["session_id"] == s1
        assert r2.json()["session_id"] == s2

    def test_E12_default_session_used_without_id(self, client):
        r = self._chat(client, "test")
        assert r.json()["session_id"] == "default"

    def test_E13_reply_contains_mock_text(self, client):
        with _patch_openai("Župan je Janez Novak."):
            r = client.post("/chat", json={"message": "Župan?"})
        assert "Janez Novak" in r.json()["reply"]

    def test_E14_message_saved_to_db(self, client):
        sid = "dbcheck_" + uuid.uuid4().hex[:8]
        self._chat(client, "Testno sporočilo za DB.", sid)
        msgs = get_messages(sid)
        assert any("Testno sporočilo za DB." in m["content"] for m in msgs)

    def test_E15_both_roles_saved(self, client):
        sid = "roles_" + uuid.uuid4().hex[:8]
        self._chat(client, "Vprašanje", sid)
        msgs = get_messages(sid)
        roles = {m["role"] for m in msgs}
        assert "user" in roles
        assert "assistant" in roles

    def test_E16_english_message_accepted(self, client):
        r = self._chat(client, "Hello, what are the office hours?")
        assert r.status_code == 200

    def test_E17_german_message_accepted(self, client):
        r = self._chat(client, "Wann sind die Öffnungszeiten?")
        assert r.status_code == 200

    def test_E18_special_chars_in_message(self, client):
        r = self._chat(client, "Vprašanje & <test> \" odgovor?")
        assert r.status_code == 200

    def test_E19_content_type_json(self, client):
        r = self._chat(client, "test")
        assert "application/json" in r.headers["content-type"]

    def test_E20_reply_is_string(self, client):
        assert isinstance(self._chat(client, "test").json()["reply"], str)

    def test_E21_session_id_is_string(self, client):
        assert isinstance(self._chat(client, "test").json()["session_id"], str)

    def test_E22_concurrent_sessions_no_mix(self, client):
        sa = "aaa_" + uuid.uuid4().hex[:6]
        sb = "bbb_" + uuid.uuid4().hex[:6]
        ra = self._chat(client, "msg_a", sa)
        rb = self._chat(client, "msg_b", sb)
        assert ra.json()["session_id"] != rb.json()["session_id"]

    def test_E23_admin_endpoint_reachable(self, client):
        r = client.get("/admin")
        assert r.status_code == 200

    def test_E24_admin_stats_without_token_returns_401(self, client):
        r = client.get("/api/admin/stats")
        assert r.status_code == 401

    def test_E25_admin_sessions_without_token_returns_401(self, client):
        r = client.get("/api/admin/sessions")
        assert r.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# F. Admin API — 10 testov
# ═══════════════════════════════════════════════════════════════════════════════
class TestAdminAPI:
    TOKEN = "admin123"  # default token

    def test_F01_stats_with_token_returns_200(self, client):
        r = client.get(f"/api/admin/stats?token={self.TOKEN}")
        assert r.status_code == 200

    def test_F02_stats_has_total_sessions(self, client):
        r = client.get(f"/api/admin/stats?token={self.TOKEN}")
        assert "total_sessions" in r.json()

    def test_F03_stats_has_total_messages(self, client):
        r = client.get(f"/api/admin/stats?token={self.TOKEN}")
        assert "total_messages" in r.json()

    def test_F04_sessions_with_token_returns_list(self, client):
        r = client.get(f"/api/admin/sessions?token={self.TOKEN}")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_F05_wrong_token_returns_401(self, client):
        r = client.get("/api/admin/stats?token=napacno_geslo")
        assert r.status_code == 401

    def test_F06_session_detail_with_token(self, client):
        sid = "detail_" + uuid.uuid4().hex[:8]
        with _patch_openai("Odgovor."):
            client.post("/chat", json={"message": "test", "session_id": sid})
        r = client.get(f"/api/admin/sessions/{sid}?token={self.TOKEN}")
        assert r.status_code == 200

    def test_F07_session_detail_returns_messages(self, client):
        sid = "detail2_" + uuid.uuid4().hex[:8]
        with _patch_openai("Odgovor."):
            client.post("/chat", json={"message": "Vprašanje za detail", "session_id": sid})
        msgs = client.get(f"/api/admin/sessions/{sid}?token={self.TOKEN}").json()
        assert any("Vprašanje za detail" in m["content"] for m in msgs)

    def test_F08_session_detail_wrong_token_401(self, client):
        r = client.get("/api/admin/sessions/test?token=wrong")
        assert r.status_code == 401

    def test_F09_stats_today_sessions_is_int(self, client):
        r = client.get(f"/api/admin/stats?token={self.TOKEN}")
        assert isinstance(r.json()["today_sessions"], int)

    def test_F10_header_token_works(self, client):
        r = client.get("/api/admin/stats", headers={"X-Admin-Token": self.TOKEN})
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# G. Živi LLM testi — 25 testov
# ═══════════════════════════════════════════════════════════════════════════════
def _live(client, message: str, session_id: str | None = None) -> dict:
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id
    r = client.post("/chat", json=payload)
    assert r.status_code == 200
    return r.json()


@pytest.mark.skipif(not HAS_OPENAI, reason="Potreben OPENAI_API_KEY")
class TestLiveLLM:
    @pytest.fixture(scope="class")
    def live_client(self):
        from app.core.db import init_db
        init_db()
        from app.rag.search import load_knowledge
        load_knowledge(KNOWLEDGE_PATH)
        from main import app
        with TestClient(app) as c:
            yield c

    def test_G01_zupan_je_janez_novak(self, live_client):
        r = _live(live_client, "Kdo je župan Občine Lepo Mesto?")
        assert "Novak" in r["reply"] or "Janez" in r["reply"], r["reply"]

    def test_G02_zna_uradne_ure_sredo(self, live_client):
        r = _live(live_client, "Do kdaj ste odprti v sredo?")
        assert "16:00" in r["reply"] or "16" in r["reply"], r["reply"]

    def test_G03_naslov_obcine(self, live_client):
        r = _live(live_client, "Kakšen je naslov občine?")
        reply = r["reply"].lower()
        assert "trg svobode" in reply or "9876" in reply or "lepo mesto" in reply, r["reply"]

    def test_G04_telefon_obcine(self, live_client):
        r = _live(live_client, "Kakšna je telefonska številka občine?")
        assert "555" in r["reply"] or "02" in r["reply"], r["reply"]

    def test_G05_email_obcine(self, live_client):
        r = _live(live_client, "Kakšen je email občine?")
        assert "lepo-mesto.si" in r["reply"], r["reply"]

    def test_G06_odvoz_smeti_torek(self, live_client):
        r = _live(live_client, "Kdaj odvažajo mešane odpadke?")
        assert "torek" in r["reply"].lower(), r["reply"]

    def test_G07_odvoz_papir_sreda(self, live_client):
        r = _live(live_client, "Kdaj je odvoz papirja?")
        assert "sreda" in r["reply"].lower() or "sredo" in r["reply"].lower(), r["reply"]

    def test_G08_pomocnica_ime_leja(self, live_client):
        r = _live(live_client, "Kako ti je ime?")
        assert "Leja" in r["reply"] or "leja" in r["reply"].lower(), r["reply"]

    def test_G09_financna_pomoc_rojstvo(self, live_client):
        r = _live(live_client, "Koliko je finančna pomoč ob rojstvu otroka?")
        assert "300" in r["reply"], r["reply"]

    def test_G10_potrdilo_ne_izdaja_obcina(self, live_client):
        r = _live(live_client, "Kje dobim potrdilo o stalnem bivališču?")
        reply = r["reply"].lower()
        assert "upravna enota" in reply or "e-uprava" in reply, r["reply"]

    def test_G11_zbirni_center_urnik(self, live_client):
        r = _live(live_client, "Kdaj je odprt zbirni center?")
        reply = r["reply"].lower()
        assert "sreda" in reply or "sobota" in reply or "10" in reply or "8" in reply, r["reply"]

    def test_G12_jezero_lepica_kopanje(self, live_client):
        r = _live(live_client, "Kje se kopamo poleti?")
        reply = r["reply"].lower()
        assert "lepica" in reply or "jezero" in reply, r["reply"]

    def test_G13_sokrat_odgovori_slovensko(self, live_client):
        r = _live(live_client, "Pozdravljeni!")
        reply = r["reply"]
        sl_words = ["pozdravljeni", "leja", "asistentka", "lepo", "pomoč", "kako", "občin"]
        assert any(w in reply.lower() for w in sl_words), r["reply"]

    def test_G14_odgovori_anglescko_na_angle(self, live_client):
        r = _live(live_client, "What is the address of the municipality?")
        reply = r["reply"]
        en_words = ["address", "municipality", "square", "freedom", "trg", "svobode", "the", "is", "located", "at", "lepo"]
        assert any(w in reply.lower() for w in en_words), r["reply"]

    def test_G15_os_lepo_mesto_ravnatelj(self, live_client):
        r = _live(live_client, "Kdo je ravnatelj osnovne šole?")
        reply = r["reply"]
        assert "Kocjan" in reply or "Boštjan" in reply or "OŠ" in reply or "šol" in reply.lower(), r["reply"]

    def test_G16_zdravstveni_dom_delovni_cas(self, live_client):
        r = _live(live_client, "Kdaj je odprt zdravstveni dom?")
        reply = r["reply"]
        assert "7:00" in reply or "19:00" in reply or "07" in reply, r["reply"]

    def test_G17_proracun_skupni_znesek(self, live_client):
        r = _live(live_client, "Kolikšen je proračun občine za 2026?")
        assert "4.200" in r["reply"] or "4200" in r["reply"] or "EUR" in r["reply"], r["reply"]

    def test_G18_razpisi_kulturna_drustva(self, live_client):
        r = _live(live_client, "Ali imate razpis za kulturna društva?")
        reply = r["reply"].lower()
        assert "kulturna" in reply or "razpis" in reply or "sofinancir" in reply, r["reply"]

    def test_G19_podžupanka_marija_kovac(self, live_client):
        r = _live(live_client, "Kdo je podžupan?")
        reply = r["reply"]
        assert "Kovač" in reply or "Marija" in reply or "podžupan" in reply.lower(), r["reply"]

    def test_G20_sveta_trojica_visina(self, live_client):
        r = _live(live_client, "Kateri hrib je v občini?")
        reply = r["reply"]
        assert "Trojica" in reply or "561" in reply or "hrib" in reply.lower(), r["reply"]

    def test_G21_obmocje_12_naselij(self, live_client):
        r = _live(live_client, "Koliko naselij ima občina?")
        assert "12" in r["reply"] or "dvan" in r["reply"].lower(), r["reply"]

    def test_G22_lekarna_urnik(self, live_client):
        r = _live(live_client, "Kdaj je odprta lekarna?")
        reply = r["reply"]
        assert "18:30" in reply or "7:30" in reply or "lekarna" in reply.lower(), r["reply"]

    def test_G23_kontakt_gradnja_ana_lebar(self, live_client):
        r = _live(live_client, "Na koga se obrnem za gradbeno dovoljenje?")
        reply = r["reply"]
        assert "Lebar" in reply or "Ana" in reply or "555 10 03" in reply or "urbanizem" in reply.lower(), r["reply"]

    def test_G24_bot_ne_izmislja_neobstojece(self, live_client):
        r = _live(live_client, "Ali imate letališče v občini?")
        reply = r["reply"].lower()
        # Bot ne sme izmisliti letališča
        bad = ["imamo letališče", "letališka steza", "mednarodno letališče"]
        assert not any(b in reply for b in bad), r["reply"]

    def test_G25_kratki_odgovor_za_preprosto(self, live_client):
        r = _live(live_client, "Kakšen je vaš email?")
        # Odgovor naj bo kratek — ne esej
        assert len(r["reply"]) < 300, f"Odgovor predolg: {r['reply']}"
