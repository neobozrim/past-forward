import json
from types import SimpleNamespace

import pytest
from PIL import Image

from sofia_harness.incremental_agent import (
    ArticleInput,
    ArticleIntegrityError,
    BLIND_SOL_READ_PROMPT,
    ChangeDecision,
    ChangeVerification,
    PROMPT,
    SOL_CHANGE_VERIFICATION_PROMPT,
    SOL_TEXT_ADJUDICATION_PROMPT,
    TARGETED_SOL_PROMPT,
    TERRA_COMPARE_PROMPT,
    TERRA_INVENTORY_PROMPT,
    IndependentInventory,
    IncrementalArticleAgent,
    IncrementalRun,
    InventoryArticle,
    InventoryAudit,
    InventoryFinding,
    VerifiedArticleLocator,
)


def tool_call(name, call_id, arguments):
    return SimpleNamespace(type="function_call", name=name, call_id=call_id, arguments=json.dumps(arguments))


class FakeResponses:
    def __init__(self):
        self.turn = 0
        self.create_calls = []
        self.parse_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        self.turn += 1
        if self.turn == 1:
            output = [tool_call("save_articles", "save-1", {"articles": [{
                "article_id": "article-1", "article_order": 1, "heading": "ЗАГЛАВИЕ",
                "verbatim_text": "ЗАГЛАВИЕ\n\nТочен текст.", "confidence": .98,
                "uncertainties": [], "source_anchor": "printed heading ЗАГЛАВИЕ",
            }]})]
        else:
            output = [tool_call("finish_page", "finish-1", {"final_scan_completed": True, "notes": "complete"})]
        return SimpleNamespace(id=f"response-{self.turn}", output=output, usage=None, output_text="")

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        if kwargs["text_format"] is IndependentInventory:
            parsed = IndependentInventory(articles=[InventoryArticle(
                inventory_id="inventory-1", article_order=1, heading="ЗАГЛАВИЕ",
                source_anchor="printed heading ЗАГЛАВИЕ", ownership_notes="one article",
            )])
        else:
            parsed = InventoryAudit(
                passed=True, all_articles_found=True, ownership_correct=True, reading_order_correct=True,
            )
        return SimpleNamespace(id=f"audit-{len(self.parse_calls)}", output_parsed=parsed, usage=None)


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


class TimeoutAfterCheckpointResponses(FakeResponses):
    def create(self, **kwargs):
        if self.turn:
            raise TimeoutError("slow page")
        return super().create(**kwargs)


class TimeoutAfterCheckpointClient:
    def __init__(self):
        self.responses = TimeoutAfterCheckpointResponses()


class MalformedAfterCheckpointResponses(FakeResponses):
    def create(self, **kwargs):
        if not self.turn:
            return super().create(**kwargs)
        self.create_calls.append(kwargs)
        self.turn += 1
        malformed = SimpleNamespace(
            type="function_call", name="save_articles", call_id="broken", arguments="{not-json",
        )
        return SimpleNamespace(id=f"response-{self.turn}", output=[malformed], usage=None, output_text="")


class MalformedAfterCheckpointClient:
    def __init__(self):
        self.responses = MalformedAfterCheckpointResponses()


class MultiArticleResponses(FakeResponses):
    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        self.turn += 1
        if self.turn <= 2:
            number = self.turn
            output = [tool_call("save_articles", f"save-{number}", {"articles": [{
                "article_id": f"article-{number}", "article_order": number, "heading": f"HEADING {number}",
                "verbatim_text": f"SECRET BODY {number}", "confidence": .98,
                "uncertainties": [], "source_anchor": f"visible heading {number}",
            }]})]
        else:
            output = [tool_call("finish_page", "finish-1", {"final_scan_completed": True, "notes": "complete"})]
        return SimpleNamespace(id=f"response-{self.turn}", output=output, usage=None, output_text="")


class MultiArticleClient:
    def __init__(self):
        self.responses = MultiArticleResponses()


class MissingArticleResponses(FakeResponses):
    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        call_number = len(self.parse_calls)
        if call_number in (1, 4):
            parsed = IndependentInventory(articles=[
                InventoryArticle(inventory_id="inventory-1", article_order=1, heading="FIRST",
                                 source_anchor="top heading", ownership_notes="first article"),
                InventoryArticle(inventory_id="inventory-2", article_order=2, heading="MISSING",
                                 source_anchor="lower heading", ownership_notes="second article"),
            ])
        elif call_number == 2:
            parsed = InventoryAudit(
                passed=False, all_articles_found=False, ownership_correct=True, reading_order_correct=True,
                findings=[InventoryFinding(
                    kind="missing_article", expected_order=2, heading="MISSING",
                    pixel_evidence="separate bold lower heading and body", required_action="read the complete lower article",
                )],
            )
        elif call_number == 3:
            parsed = ArticleInput(
                article_id="article-2", article_order=2, heading="MISSING", verbatim_text="RECOVERED BODY",
                confidence=.97, source_anchor="lower heading",
            )
        else:
            parsed = InventoryAudit(
                passed=True, all_articles_found=True, ownership_correct=True, reading_order_correct=True,
            )
        return SimpleNamespace(id=f"parse-{call_number}", output_parsed=parsed, usage=None)


class MissingArticleClient:
    def __init__(self):
        self.responses = MissingArticleResponses()


class FinishResponses(FakeResponses):
    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        self.turn += 1
        output = [tool_call("finish_page", "finish-resume", {"final_scan_completed": True, "notes": "complete"})]
        return SimpleNamespace(id=f"finish-{self.turn}", output=output, usage=None, output_text="")


class FinishClient:
    def __init__(self):
        self.responses = FinishResponses()


class FindingOnEveryAuditResponses(FakeResponses):
    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        call_number = len(self.parse_calls)
        if call_number in (1, 4):
            parsed = IndependentInventory(articles=[InventoryArticle(
                inventory_id="inventory-1", article_order=1, heading="ЗАГЛАВИЕ",
                source_anchor="top heading", ownership_notes="one article",
            )])
        elif call_number in (2, 5):
            parsed = InventoryAudit(
                passed=False, all_articles_found=True, ownership_correct=False, reading_order_correct=True,
                findings=[InventoryFinding(
                    kind="ownership", related_article_id="article-1", expected_order=1,
                    heading="ЗАГЛАВИЕ", pixel_evidence="visible boundary mismatch",
                    required_action=f"inspect ownership pass {call_number}",
                )],
            )
        else:
            parsed = ArticleInput(
                article_id="ignored-by-revision", article_order=1, heading="ЗАГЛАВИЕ",
                verbatim_text=f"REPAIRED BODY {call_number}", confidence=.97,
                source_anchor="top heading",
            )
        return SimpleNamespace(id=f"parse-{call_number}", output_parsed=parsed, usage=None)


class FindingOnEveryAuditClient:
    def __init__(self):
        self.responses = FindingOnEveryAuditResponses()


class TextVerificationResponses(FakeResponses):
    def __init__(self, blind_text="ПЛАЧИ народът.", fail_blind=False,
                 blind_confidence=.99, final_confidence=.99, fail_adjudication=False,
                 change_choice="adjudicated", change_confidence=.99,
                 fail_change_verification=False):
        super().__init__()
        self.blind_text = blind_text
        self.fail_blind = fail_blind
        self.blind_confidence = blind_confidence
        self.final_confidence = final_confidence
        self.fail_adjudication = fail_adjudication
        self.change_choice = change_choice
        self.change_confidence = change_confidence
        self.fail_change_verification = fail_change_verification
        self.article_reads = 0

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        self.turn += 1
        if self.turn == 1:
            output = [tool_call("save_articles", "save-1", {"articles": [{
                "article_id": "article-1", "article_order": 1, "heading": "ЗАГЛАВИЕ",
                "verbatim_text": "ПЛАЧЕ народът.", "confidence": .96,
                "uncertainties": [], "source_anchor": "SECRET_READER_A_ANCHOR ПЛАЧЕ народът.",
            }]})]
        else:
            output = [tool_call("finish_page", "finish-1", {
                "final_scan_completed": True, "notes": "complete",
            })]
        return SimpleNamespace(id=f"response-{self.turn}", output=output, usage=None, output_text="")

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        if kwargs["text_format"] is IndependentInventory:
            parsed = IndependentInventory(articles=[InventoryArticle(
                inventory_id="inventory-1", article_order=1, heading="ЗАГЛАВИЕ",
                source_anchor="printed heading ЗАГЛАВИЕ", ownership_notes="one article",
            )])
        elif kwargs["text_format"] is InventoryAudit:
            parsed = InventoryAudit(
                passed=True, all_articles_found=True, ownership_correct=True, reading_order_correct=True,
                article_locators=[VerifiedArticleLocator(
                    article_id="article-1", article_order=1,
                    independently_read_heading="ЗАГЛАВИЕ",
                    visual_anchor="independently located bold heading at top",
                    ownership_notes="one independently located article",
                )],
            )
        elif kwargs["text_format"] is ChangeVerification:
            if self.fail_change_verification:
                raise TimeoutError("change verifier timed out")
            payload = json.loads(kwargs["input"][0]["content"][0]["text"])
            parsed = ChangeVerification(
                all_changes_checked=True,
                decisions=[ChangeDecision(
                    change_id=change["change_id"], choice=self.change_choice,
                    confidence=self.change_confidence,
                    pixel_evidence="printed candidate is visible",
                ) for change in payload["proposed_changes"]],
            )
        else:
            is_blind = kwargs["instructions"] == BLIND_SOL_READ_PROMPT
            self.article_reads += 1
            if is_blind and self.fail_blind:
                raise TimeoutError("blind reader timed out")
            if kwargs["instructions"] == SOL_TEXT_ADJUDICATION_PROMPT and self.fail_adjudication:
                raise TimeoutError("adjudicator timed out")
            text = self.blind_text if is_blind else "ПЛАЧИ народът."
            confidence = self.blind_confidence if is_blind else self.final_confidence
            parsed = ArticleInput(
                article_id="different-worker-id", article_order=99, heading="ЗАГЛАВИЕ",
                verbatim_text=text, confidence=confidence, source_anchor="worker anchor",
            )
        return SimpleNamespace(id=f"parse-{len(self.parse_calls)}", output_parsed=parsed, usage=None)


class TextVerificationClient:
    def __init__(self, blind_text="ПЛАЧИ народът.", fail_blind=False,
                 blind_confidence=.99, final_confidence=.99, fail_adjudication=False,
                 change_choice="adjudicated", change_confidence=.99,
                 fail_change_verification=False):
        self.responses = TextVerificationResponses(
            blind_text, fail_blind, blind_confidence, final_confidence, fail_adjudication,
            change_choice, change_confidence, fail_change_verification)


class ResumeTextVerificationResponses(TextVerificationResponses):
    def create(self, **kwargs):
        self.create_calls.append(kwargs);self.turn += 1
        output = [tool_call("finish_page", "finish-resume", {
            "final_scan_completed": True, "notes": "resume verification",
        })]
        return SimpleNamespace(id=f"resume-{self.turn}", output=output, usage=None, output_text="")


class ResumeTextVerificationClient:
    def __init__(self):
        self.responses = ResumeTextVerificationResponses()


def test_incremental_agent_checkpoints_articles_without_layout(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    events = []
    result_client = FakeClient()
    result = IncrementalArticleAgent(tmp_path / "digitized", result_client).run(
        source, "digitise the complete page", lambda stage, detail="": events.append((stage, detail)),
        verify_text=False)
    assert result["workflow"] == "incremental_full_page"
    assert result["article_count"] == 1 and result["region_count"] == 0 and result["verified"] is True
    assert result["overlay_path"] is None
    markdown = open(result["markdown_path"], encoding="utf-8").read()
    assert "## ЗАГЛАВИЕ" in markdown and "Точен текст." in markdown
    assert markdown.count("ЗАГЛАВИЕ") == 1
    state = json.loads(open(result["layout_path"], encoding="utf-8").read())
    assert state["status"] == "complete" and len(state["events"]) == 1
    assert [call["model"] for call in result_client.responses.parse_calls] == ["gpt-5.6-terra", "gpt-5.6-terra"]
    blind_inventory_text = result_client.responses.parse_calls[0]["input"][0]["content"][0]["text"]
    comparison_text = result_client.responses.parse_calls[1]["input"][0]["content"][0]["text"]
    assert "Точен текст" not in blind_inventory_text and "Точен текст" not in comparison_text
    assert any(stage == "Article saved" for stage, _ in events)


def test_incremental_articles_include_optional_filename_metadata(tmp_path):
    source = tmp_path / "IMG_0982_nm_07_04_1949_page1.png"
    Image.new("RGB", (160, 240), "white").save(source)
    result = IncrementalArticleAgent(tmp_path / "digitized", FakeClient()).run(
        source, "digitise the complete page", verify_text=False)
    payload = json.loads(open(result["articles_path"], encoding="utf-8").read())
    assert payload["metadata"] == {
        "publication": "Народна младеж", "publication_code": "nm",
        "issue_date": "1949-07-04", "page_number": 1,
    }


def test_timeout_returns_persisted_partial_articles(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    result = IncrementalArticleAgent(tmp_path / "digitized", TimeoutAfterCheckpointClient()).run(
        source, "digitise the complete page", max_context_restarts=0)
    assert result["partial"] is True and result["status"] == "interrupted" and result["article_count"] == 1


def test_malformed_model_tool_call_preserves_checkpoint_and_returns_interrupted(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    result = IncrementalArticleAgent(tmp_path / "digitized", MalformedAfterCheckpointClient()).run(
        source, "digitise the complete page", verify_text=False)
    assert result["partial"] is True and result["status"] == "interrupted"
    assert result["article_count"] == 1
    state = json.loads(open(result["layout_path"], encoding="utf-8").read())
    assert state["error"].startswith("invalid save_articles call")


def test_every_vision_worker_explicitly_forbids_external_ocr():
    for prompt in (PROMPT, TERRA_INVENTORY_PROMPT, TERRA_COMPARE_PROMPT, TARGETED_SOL_PROMPT,
                   BLIND_SOL_READ_PROMPT, SOL_TEXT_ADJUDICATION_PROMPT,
                   SOL_CHANGE_VERIFICATION_PROMPT):
        assert "Tesseract" in prompt
        assert "external OCR" in prompt


def test_sol_context_rotation_carries_anchors_but_not_prior_transcription(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    client = MultiArticleClient()
    result = IncrementalArticleAgent(tmp_path / "digitized", client).run(
        source, "digitise the complete page", max_context_turns=1, verify_text=False)
    assert result["verified"] is True and result["article_count"] == 2
    second_request = client.responses.create_calls[1]
    second_text = second_request["input"][0]["content"][0]["text"]
    assert "previous_response_id" not in second_request
    assert "HEADING 1" in second_text and "SECRET BODY 1" not in second_text
    assert "SECRET BODY 2" in open(result["markdown_path"], encoding="utf-8").read()


def test_terra_missing_inventory_routes_to_fresh_targeted_sol(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    client = MissingArticleClient()
    result = IncrementalArticleAgent(tmp_path / "digitized", client).run(
        source, "digitise the complete page", verify_text=False)
    assert result["verified"] is True and result["article_count"] == 2
    assert result["omissions_found"] == ["MISSING"]
    targeted = client.responses.parse_calls[2]
    assert targeted["model"] == "gpt-5.6-sol"
    assert targeted["text_format"] is ArticleInput
    targeted_text = targeted["input"][0]["content"][0]["text"]
    assert "RECOVERED BODY" not in targeted_text
    assert "MISSING" in targeted_text


def test_every_terra_finding_routes_to_sol_even_on_final_audit(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    client = FindingOnEveryAuditClient()
    result = IncrementalArticleAgent(tmp_path / "digitized", client).run(
        source, "digitise the complete page", max_audits=2, verify_text=False)
    targeted = [call for call in client.responses.parse_calls if call["text_format"] is ArticleInput]
    assert len(targeted) == 2
    assert result["status"] == "repaired_pending_inventory_confirmation"
    assert result["verified"] is False and result["partial"] is True


def test_blind_sol_read_and_fresh_adjudication_repair_text_automatically(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    events = []
    client = TextVerificationClient()
    result = IncrementalArticleAgent(tmp_path / "digitized", client).run(
        source, "digitise the complete page", lambda stage, detail="": events.append((stage, detail)))
    assert result["verified"] is True and result["text_verified_articles"] == ["article-1"]
    markdown = open(result["markdown_path"], encoding="utf-8").read()
    assert "ПЛАЧИ народът." in markdown and "ПЛАЧЕ народът." not in markdown
    blind_call = client.responses.parse_calls[2]
    blind_input = blind_call["input"][0]["content"][0]["text"]
    assert blind_call["model"] == "gpt-5.6-sol"
    assert "ПЛАЧЕ народът" not in blind_input
    assert "SECRET_READER_A_ANCHOR" not in blind_input
    adjudication_call = client.responses.parse_calls[3]
    adjudication_input = adjudication_call["input"][0]["content"][0]["text"]
    assert "ПЛАЧЕ народът" in adjudication_input and "ПЛАЧИ народът" in adjudication_input
    change_call = client.responses.parse_calls[4]
    assert change_call["instructions"] == SOL_CHANGE_VERIFICATION_PROMPT
    change_input = change_call["input"][0]["content"][0]["text"]
    assert "ПЛАЧЕ" in change_input and "ПЛАЧИ" in change_input
    assert "reader_b" not in change_input
    state = json.loads(open(result["layout_path"], encoding="utf-8").read())
    record = state["text_verification"]["article-1"]
    assert record["status"] == "adjudicated"
    assert open(record["blind_read_path"], encoding="utf-8").read()
    assert open(record["adjudication_path"], encoding="utf-8").read()
    assert open(record["change_verification_path"], encoding="utf-8").read()
    assert any(stage == "Text disagreement resolved" for stage, _ in events)


def test_matching_blind_read_skips_adjudicator(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    client = TextVerificationClient(blind_text="ПЛАЧЕ   народът.")
    result = IncrementalArticleAgent(tmp_path / "digitized", client).run(source, "digitise")
    article_calls = [call for call in client.responses.parse_calls if call["text_format"] is ArticleInput]
    assert len(article_calls) == 1
    assert result["verified"] is True
    state = json.loads(open(result["layout_path"], encoding="utf-8").read())
    assert state["text_verification"]["article-1"]["status"] == "matched"


def test_source_change_verifier_can_reject_confident_adjudicator_edit(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    client = TextVerificationClient(change_choice="reader_a")
    result = IncrementalArticleAgent(tmp_path / "digitized", client).run(source, "digitise")
    assert result["verified"] is True
    markdown = open(result["markdown_path"], encoding="utf-8").read()
    assert "ПЛАЧЕ народът." in markdown and "ПЛАЧИ народът." not in markdown
    state = json.loads(open(result["layout_path"], encoding="utf-8").read())
    record = state["text_verification"]["article-1"]
    artifact = json.loads(open(record["change_verification_path"], encoding="utf-8").read())
    assert artifact["verification"]["decisions"][0]["choice"] == "reader_a"


def test_change_verifier_failure_preserves_primary_and_blocks_completion(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    client = TextVerificationClient(fail_change_verification=True)
    result = IncrementalArticleAgent(tmp_path / "digitized", client).run(source, "digitise")
    assert result["verified"] is False and result["status"] == "text_review_required"
    assert "ПЛАЧЕ народът." in open(result["markdown_path"], encoding="utf-8").read()
    state = json.loads(open(result["layout_path"], encoding="utf-8").read())
    assert state["text_verification"]["article-1"]["status"] == "change_verification_failed"


def test_blind_read_failure_preserves_primary_and_never_claims_complete(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    result = IncrementalArticleAgent(tmp_path / "digitized", TextVerificationClient(fail_blind=True)).run(
        source, "digitise")
    assert result["status"] == "text_review_required" and result["verified"] is False
    assert result["text_verification_pending"] == ["article-1"]
    assert "ПЛАЧЕ народът." in open(result["markdown_path"], encoding="utf-8").read()


def test_low_confidence_adjudication_cannot_replace_primary_or_complete(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    result = IncrementalArticleAgent(
        tmp_path / "digitized", TextVerificationClient(final_confidence=.4)).run(source, "digitise")
    assert result["status"] == "text_review_required" and result["verified"] is False
    assert "ПЛАЧЕ народът." in open(result["markdown_path"], encoding="utf-8").read()
    state = json.loads(open(result["layout_path"], encoding="utf-8").read())
    assert state["text_verification"]["article-1"]["status"] == "adjudication_low_confidence"


def test_comparison_preserves_line_and_paragraph_disagreements():
    one_line = IncrementalArticleAgent._comparison("първи втори", "първи   втори")
    line_break = IncrementalArticleAgent._comparison("първи втори", "първи\nвтори")
    paragraph = IncrementalArticleAgent._comparison("първи\nвтори", "първи\n\nвтори")
    assert one_line.lexically_equal is True
    assert line_break.lexically_equal is False and line_break.difference_count >= 1
    assert paragraph.lexically_equal is False and paragraph.difference_count >= 1


def test_read_artifacts_cannot_collide_after_filename_sanitization(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    run = IncrementalRun(tmp_path / "digitized", source, "digitise")
    first = run.save_read_artifact("a/b", "sol-b", {"article": "first"})
    second = run.save_read_artifact("a?b", "sol-b", {"article": "second"})
    assert first != second and first.is_file() and second.is_file()
    assert json.loads(first.read_text(encoding="utf-8"))["article"] == "first"


def test_resume_refuses_changed_archived_source(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    partial = IncrementalArticleAgent(tmp_path / "digitized", TimeoutAfterCheckpointClient()).run(
        source, "digitise", max_context_restarts=0)
    Image.new("RGB", (160, 240), "black").save(partial["source_path"])
    try:
        IncrementalRun(tmp_path / "digitized", partial["source_path"], "continue",
                       resume_folder=partial["folder"])
    except ValueError as exc:
        assert "source hash changed" in str(exc)
    else:
        raise AssertionError("changed source was accepted")


def test_resume_reuses_saved_blind_read_after_adjudication_timeout(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    partial_client = TextVerificationClient(fail_adjudication=True)
    partial = IncrementalArticleAgent(tmp_path / "digitized", partial_client).run(source, "digitise")
    assert partial["status"] == "text_review_required"
    partial_state = json.loads(open(partial["layout_path"], encoding="utf-8").read())
    blind_path = partial_state["text_verification"]["article-1"]["blind_read_path"]

    resumed_client = ResumeTextVerificationClient()
    resumed = IncrementalArticleAgent(tmp_path / "digitized", resumed_client).run(
        partial["source_path"], "continue", resume_folder=partial["folder"])
    article_calls = [call for call in resumed_client.responses.parse_calls if call["text_format"] is ArticleInput]
    assert len(article_calls) == 1
    assert article_calls[0]["instructions"] == SOL_TEXT_ADJUDICATION_PROMPT
    assert resumed["verified"] is True
    resumed_state = json.loads(open(resumed["layout_path"], encoding="utf-8").read())
    assert resumed_state["text_verification"]["article-1"]["blind_read_path"] == blind_path
    assert "ПЛАЧИ народът." in open(resumed["markdown_path"], encoding="utf-8").read()


def test_interrupted_incremental_run_resumes_without_repeating_saved_article(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    partial = IncrementalArticleAgent(tmp_path / "digitized", TimeoutAfterCheckpointClient()).run(
        source, "digitise", max_context_restarts=0)
    resumed = IncrementalArticleAgent(tmp_path / "digitized", FinishClient()).run(
        partial["source_path"], "continue", resume_folder=partial["folder"], verify_text=False)
    assert resumed["folder"] == partial["folder"]
    assert resumed["verified"] is True and resumed["article_count"] == 1
    state = json.loads(open(resumed["layout_path"], encoding="utf-8").read())
    assert any(event["event"] == "resumed" for event in state["events"])


def test_targeted_revision_can_reorder_articles_without_order_collisions(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    run = IncrementalRun(tmp_path / "digitized", source, "digitise")
    run.save(ArticleInput(article_id="first", article_order=1, heading="FIRST",
                          verbatim_text="one", confidence=.9, source_anchor="top"))
    run.save(ArticleInput(article_id="second", article_order=2, heading="SECOND",
                          verbatim_text="two", confidence=.9, source_anchor="lower"))
    run.revise(ArticleInput(article_id="second", article_order=1, heading="SECOND",
                            verbatim_text="two", confidence=.9, source_anchor="lower"), "Terra order finding")
    rows = json.loads(run.articles_path.read_text(encoding="utf-8"))["articles"]
    assert [(row["article_id"], row["article_order"]) for row in rows] == [("second", 1), ("first", 2)]


def test_internal_truncation_marker_can_never_be_checkpointed(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    run = IncrementalRun(tmp_path / "digitized", source, "digitise")
    corrupted = ArticleInput(
        article_id="broken", article_order=1, heading="ARTICLE",
        verbatim_text="Visible beginning …3778 tokens truncated… unrelated tail",
        confidence=.99, source_anchor="starts at ARTICLE and ends at a byline",
    )
    with pytest.raises(ArticleIntegrityError, match="output-truncation"):
        run.save(corrupted)
    assert run.result()["article_count"] == 0


def test_duplicate_source_anchor_is_rejected(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    run = IncrementalRun(tmp_path / "digitized", source, "digitise")
    run.save(ArticleInput(article_id="one", article_order=1, heading="ONE", verbatim_text="first",
                          confidence=.9, source_anchor="top left through the Иванов byline"))
    with pytest.raises(ArticleIntegrityError, match="source anchor duplicates"):
        run.save(ArticleInput(article_id="two", article_order=2, heading="TWO", verbatim_text="second",
                              confidence=.9, source_anchor=" top  left through the Иванов BYLINE "))


def test_long_boundary_text_owned_by_two_articles_is_rejected(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    run = IncrementalRun(tmp_path / "digitized", source, "digitise")
    shared = " ".join(f"word{index}" for index in range(80))
    run.save(ArticleInput(article_id="one", article_order=1, heading="ONE",
                          verbatim_text="unique opening " + shared, confidence=.9,
                          source_anchor="upper article ending at its signature"))
    with pytest.raises(ArticleIntegrityError, match="boundary text overlap"):
        run.save(ArticleInput(article_id="two", article_order=2, heading="TWO",
                              verbatim_text=shared + " unique ending", confidence=.9,
                              source_anchor="lower article ending at its rule"))


def test_incomplete_model_response_is_rejected_before_parsing():
    response = SimpleNamespace(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    with pytest.raises(ArticleIntegrityError, match="max_output_tokens"):
        IncrementalArticleAgent._require_complete_response(response, "Sol article transcription")


def test_rejected_integrity_revision_does_not_change_article_order(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    run = IncrementalRun(tmp_path / "digitized", source, "digitise")
    run.save(ArticleInput(article_id="one", article_order=1, heading="ONE", verbatim_text="first",
                          confidence=.9, source_anchor="upper article through first byline"))
    run.save(ArticleInput(article_id="two", article_order=2, heading="TWO", verbatim_text="second",
                          confidence=.9, source_anchor="lower article through second byline"))
    corrupt_revision = ArticleInput(
        article_id="two", article_order=1, heading="TWO",
        verbatim_text="second …500 tokens truncated… tail", confidence=.9,
        source_anchor="lower article through second byline",
    )
    with pytest.raises(ArticleIntegrityError):
        run.revise(corrupt_revision, "bad repair")
    assert [(row["article_id"], row["article_order"])
            for row in sorted(run.state["articles"].values(), key=lambda row: row["article_order"])] == [
        ("one", 1), ("two", 2),
    ]


def test_truncation_signals_find_abrupt_end_and_missing_suffix():
    abrupt = IncrementalArticleAgent._truncation_signals("Текстът свършва по-", "Текстът свършва по-")
    longer_b = IncrementalArticleAgent._truncation_signals(
        "Началото на статията е тук.",
        "Началото на статията е тук. " + " ".join(f"добавена{index}" for index in range(10)),
    )
    assert any("mid-word" in signal for signal in abrupt)
    assert any("additional ending words" in signal for signal in longer_b)


def test_missing_ending_triggers_existing_source_adjudication_path(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (160, 240), "white").save(source)
    blind = "ПЛАЧЕ народът. " + " ".join(f"добавена{index}" for index in range(10))
    events = []
    result = IncrementalArticleAgent(
        tmp_path / "digitized", TextVerificationClient(blind_text=blind)).run(
            source, "digitise", progress=lambda stage, detail="": events.append((stage, detail)))
    state = json.loads(open(result["layout_path"], encoding="utf-8").read())
    review = state["text_verification"]["article-1"]["truncation_review"]
    assert any(stage == "Possible truncation detected" for stage, _ in events)
    assert review["status"] == "resolved_by_source_adjudication"
    assert result["verified"] is True
